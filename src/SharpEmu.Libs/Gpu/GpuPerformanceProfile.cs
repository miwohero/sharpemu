// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Libs.Gpu;

/// <summary>
/// Perfis de performance adaptativos para diferentes classes de GPUs.
/// Detecta automaticamente a capacidade da GPU e ajusta parâmetros
/// para balancear qualidade visual e performance.
///
/// GPUs suportadas:
/// - Entry Level: RTX 3050 (4GB/8GB), RX 5700 (8GB), GTX 1650, etc.
/// - Mid Range: RTX 3060/4060, RX 6600/7600, etc.
/// - High End: RTX 4070+, RX 7800+, etc.
/// </summary>
public sealed class GpuPerformanceProfile
{
    public enum GpuTier
    {
        Unknown,
        EntryLevel,    // RTX 3050, RX 5700, GTX 1650
        MidRange,      // RTX 3060/4060, RX 6600
        HighEnd,       // RTX 4070+, RX 7800+
    }

    public GpuTier Tier { get; }
    public int RecommendedResolutionScale { get; }
    public bool UseAsyncCompute { get; }
    public int MaxTextureCacheSizeMB { get; }
    public int MaxBufferPoolSizeMB { get; }
    public bool PreferCpuDetileForSmallTextures { get; }
    public int MaxConcurrentUploads { get; }
    public bool EnableShaderCache { get; }
    public int ShaderCacheSize { get; }
    public bool UseComputeDetile { get; }
    public int FrameLatencyTargetMs { get; }
    public bool AggressiveGarbageCollection { get; }

    private GpuPerformanceProfile(
        GpuTier tier,
        int resolutionScale,
        bool asyncCompute,
        int textureCacheMB,
        int bufferPoolMB,
        bool cpuDetileSmall,
        int maxUploads,
        bool shaderCache,
        int cacheSize,
        bool computeDetile,
        int latencyMs,
        bool aggressiveGC)
    {
        Tier = tier;
        RecommendedResolutionScale = resolutionScale;
        UseAsyncCompute = asyncCompute;
        MaxTextureCacheSizeMB = textureCacheMB;
        MaxBufferPoolSizeMB = bufferPoolMB;
        PreferCpuDetileForSmallTextures = cpuDetileSmall;
        MaxConcurrentUploads = maxUploads;
        EnableShaderCache = shaderCache;
        ShaderCacheSize = cacheSize;
        UseComputeDetile = computeDetile;
        FrameLatencyTargetMs = latencyMs;
        AggressiveGarbageCollection = aggressiveGC;
    }

    /// <summary>
    /// Perfil otimizado para GPUs de entrada como RTX 3050 e RX 5700.
    /// Prioriza performance sobre qualidade visual máxima.
    /// </summary>
    public static GpuPerformanceProfile EntryLevel { get; } = new(
        tier: GpuTier.EntryLevel,
        resolutionScale: 75,              // Renderiza a 75% da resolução nativa
        asyncCompute: false,              // Desabilita async compute para evitar stalls
        textureCacheMB: 256,              // Cache de texturas conservador
        bufferPoolMB: 64,                 // Pool de buffers reduzido
        cpuDetileSmall: true,             // Usa CPU para texturas pequenas (< 256x256)
        maxUploads: 2,                    // Limita uploads concorrentes
        shaderCache: true,                // Cache de shaders ativado
        cacheSize: 256,                   // Cache de 256 shaders
        computeDetile: true,              // Usa compute shader para detile
        latencyMs: 33,                    // Target: 30 FPS mínimo
        aggressiveGC: true                // GC mais agressivo para liberar VRAM
    );

    /// <summary>
    /// Perfil para GPUs mid-range como RTX 3060/4060 e RX 6600.
    /// </summary>
    public static GpuPerformanceProfile MidRange { get; } = new(
        tier: GpuTier.MidRange,
        resolutionScale: 90,
        asyncCompute: true,
        textureCacheMB: 512,
        bufferPoolMB: 128,
        cpuDetileSmall: false,
        maxUploads: 4,
        shaderCache: true,
        cacheSize: 512,
        computeDetile: true,
        latencyMs: 16,
        aggressiveGC: false
    );

    /// <summary>
    /// Perfil para GPUs high-end como RTX 4070+ e RX 7800+.
    /// </summary>
    public static GpuPerformanceProfile HighEnd { get; } = new(
        tier: GpuTier.HighEnd,
        resolutionScale: 100,
        asyncCompute: true,
        textureCacheMB: 1024,
        bufferPoolMB: 256,
        cpuDetileSmall: false,
        maxUploads: 8,
        shaderCache: true,
        cacheSize: 1024,
        computeDetile: true,
        latencyMs: 8,
        aggressiveGC: false
    );

    /// <summary>
    /// Detecta o tier da GPU baseado no nome do dispositivo Vulkan.
    /// </summary>
    public static GpuTier DetectGpuTier(string deviceName)
    {
        if (string.IsNullOrEmpty(deviceName))
            return GpuTier.Unknown;

        var name = deviceName.ToLowerInvariant();

        // Entry Level detection
        if (name.Contains("3050") || name.Contains("3040") ||
            name.Contains("1650") || name.Contains("1630") ||
            name.Contains("1050") || name.Contains("1030") ||
            name.Contains("rx 5700") || name.Contains("rx 5600") ||
            name.Contains("rx 5500") || name.Contains("rx 6500") ||
            name.Contains("rx 6400") || name.Contains("arc a380") ||
            name.Contains("arc a310"))
        {
            return GpuTier.EntryLevel;
        }

        // High End detection
        if (name.Contains("4090") || name.Contains("4080") || name.Contains("4070") ||
            name.Contains("3090") || name.Contains("3080") || name.Contains("3070") ||
            name.Contains("rx 7900") || name.Contains("rx 7800") ||
            name.Contains("rx 6900") || name.Contains("rx 6800") ||
            name.Contains("arc a770") || name.Contains("arc a750"))
        {
            return GpuTier.HighEnd;
        }

        // Mid Range (default)
        return GpuTier.MidRange;
    }

    /// <summary>
    /// Retorna o perfil apropriado para o tier detectado.
    /// </summary>
    public static GpuPerformanceProfile GetProfile(GpuTier tier) => tier switch
    {
        GpuTier.EntryLevel => EntryLevel,
        GpuTier.HighEnd => HighEnd,
        _ => MidRange,
    };

    /// <summary>
    /// Retorna o perfil apropriado baseado no nome do dispositivo Vulkan.
    /// </summary>
    public static GpuPerformanceProfile GetProfileForDevice(string deviceName)
    {
        var tier = DetectGpuTier(deviceName);
        return GetProfile(tier);
    }

    public override string ToString() =>
        $"GpuPerformanceProfile[Tier={Tier}, Scale={RecommendedResolutionScale}%, " +
        $"AsyncCompute={UseAsyncCompute}, Cache={MaxTextureCacheSizeMB}MB, " +
        $"LatencyTarget={FrameLatencyTargetMs}ms]";
}
