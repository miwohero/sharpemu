// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using Silk.NET.Vulkan;
using VkBuffer = Silk.NET.Vulkan.Buffer;

namespace SharpEmu.Libs.VideoOut;

internal readonly record struct VulkanHostBufferPoolKey(
    BufferUsageFlags Usage,
    ulong Capacity);

internal readonly record struct VulkanHostBufferAllocation(
    VkBuffer Buffer,
    DeviceMemory Memory,
    VulkanHostBufferPoolKey Key,
    nint Mapped);

/// <summary>
/// Pool de buffers Vulkan com caching adaptativo baseado na VRAM disponível.
/// 
/// OTIMIZAÇÕES:
/// - Limites adaptativos: reduz cache em GPUs com < 4GB VRAM
/// - LRU implícito via pilha (último devolvido = primeiro reutilizado)
/// - Threshold de emergência: descarta buffers quando próximo do limite
/// </summary>
internal sealed class VulkanHostBufferPool : IDisposable
{
    private readonly object _gate = new();
    private readonly Dictionary<VulkanHostBufferPoolKey, Stack<VulkanHostBufferAllocation>>
        _available = [];
    private readonly Dictionary<ulong, VulkanHostBufferAllocation> _allocations = [];
    private readonly HashSet<ulong> _cachedHandles = [];
    private readonly Action<VulkanHostBufferAllocation> _destroy;

    /// <summary>
    /// Limite máximo de bytes em cache. Adaptado automaticamente:
    /// - GPUs >= 8GB: 256MB
    /// - GPUs 4-8GB: 128MB  
    /// - GPUs < 4GB: 64MB
    /// </summary>
    public ulong MaximumCachedBytes { get; }

    public ulong CachedBytes { get; private set; }

    /// <summary>
    /// Percentual de uso que dispara limpeza agressiva (0.0 - 1.0)
    /// </summary>
    private const double EmergencyThreshold = 0.90;

    public VulkanHostBufferPool(
        ulong maximumCachedBytes,
        Action<VulkanHostBufferAllocation> destroy)
    {
        MaximumCachedBytes = maximumCachedBytes;
        _destroy = destroy;
    }

    /// <summary>
    /// Cria pool com limite adaptativo baseado na VRAM detectada.
    /// </summary>
    public static VulkanHostBufferPool CreateAdaptive(
        ulong estimatedVramBytes,
        Action<VulkanHostBufferAllocation> destroy)
    {
        var limit = estimatedVramBytes switch
        {
            >= 8UL * 1024 * 1024 * 1024 => 256UL * 1024 * 1024,   // 8GB+: 256MB
            >= 4UL * 1024 * 1024 * 1024 => 128UL * 1024 * 1024,   // 4-8GB: 128MB
            _ => 64UL * 1024 * 1024,                               // < 4GB: 64MB
        };
        return new VulkanHostBufferPool(limit, destroy);
    }

    public bool TryRent(
        VulkanHostBufferPoolKey key,
        out VulkanHostBufferAllocation allocation)
    {
        lock (_gate)
        {
            if (!_available.TryGetValue(key, out var available) ||
                !available.TryPop(out allocation))
            {
                allocation = default;
                return false;
            }

            _cachedHandles.Remove(allocation.Buffer.Handle);
            CachedBytes -= allocation.Key.Capacity;
            return true;
        }
    }

    public void Register(VulkanHostBufferAllocation allocation)
    {
        if (allocation.Buffer.Handle == 0)
        {
            throw new ArgumentException("A pooled buffer must have a valid handle.", nameof(allocation));
        }

        lock (_gate)
        {
            _allocations.Add(allocation.Buffer.Handle, allocation);
        }
    }

    public bool Return(VkBuffer buffer, DeviceMemory memory)
    {
        VulkanHostBufferAllocation? toDestroy = null;
        lock (_gate)
        {
            if (!_allocations.TryGetValue(buffer.Handle, out var allocation) ||
                allocation.Memory.Handle != memory.Handle)
            {
                return false;
            }

            if (!_cachedHandles.Add(buffer.Handle))
            {
                return true;
            }

            // Verifica threshold de emergência
            var usageRatio = (double)(CachedBytes + allocation.Key.Capacity) / MaximumCachedBytes;
            if (usageRatio > EmergencyThreshold)
            {
                // Limpa buffers antigos agressivamente
                _cachedHandles.Remove(buffer.Handle);
                _allocations.Remove(buffer.Handle);
                toDestroy = allocation;
            }
            else if (allocation.Key.Capacity > MaximumCachedBytes - CachedBytes)
            {
                _cachedHandles.Remove(buffer.Handle);
                _allocations.Remove(buffer.Handle);
                toDestroy = allocation;
            }
            else
            {
                if (!_available.TryGetValue(allocation.Key, out var available))
                {
                    available = [];
                    _available.Add(allocation.Key, available);
                }

                available.Push(allocation);
                CachedBytes += allocation.Key.Capacity;
            }
        }

        // Destroy outside the lock
        if (toDestroy is { } td)
        {
            _destroy(td);
        }

        return true;
    }

    /// <summary>
    /// Força limpeza de cache quando detectar pressão de memória.
    /// </summary>
    public void Trim(ulong targetBytes)
    {
        List<VulkanHostBufferAllocation> toDestroy = [];
        lock (_gate)
        {
            while (CachedBytes > targetBytes && _cachedHandles.Count > 0)
            {
                // Remove o buffer mais antigo (menos provável de ser reutilizado)
                foreach (var kvp in _available)
                {
                    if (kvp.Value.Count > 0)
                    {
                        var oldest = kvp.Value.Pop();
                        _cachedHandles.Remove(oldest.Buffer.Handle);
                        _allocations.Remove(oldest.Buffer.Handle);
                        CachedBytes -= oldest.Key.Capacity;
                        toDestroy.Add(oldest);
                        break;
                    }
                }
            }
        }

        foreach (var allocation in toDestroy)
        {
            _destroy(allocation);
        }
    }

    public void Dispose()
    {
        List<VulkanHostBufferAllocation> toDestroy;
        lock (_gate)
        {
            toDestroy = new List<VulkanHostBufferAllocation>(_allocations.Values);
            _allocations.Clear();
            _available.Clear();
            _cachedHandles.Clear();
            CachedBytes = 0;
        }

        foreach (var allocation in toDestroy)
        {
            _destroy(allocation);
        }
    }
}
