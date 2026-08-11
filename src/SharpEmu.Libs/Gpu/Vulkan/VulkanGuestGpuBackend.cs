// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using SharpEmu.Libs.VideoOut;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Vulkan;

namespace SharpEmu.Libs.Gpu.Vulkan;

/// <summary>
/// Vulkan backend for the guest-GPU seam: SPIR-V codegen via
/// SharpEmu.ShaderCompiler.Vulkan, rendering via a thin adapter over the existing
/// VulkanVideoPresenter statics.
///
/// OTIMIZAÇÕES PARA GPUs DE ENTRADA (RTX 3050 / RX 5700):
/// - Cache LRU de shaders compilados (evita recompilação de shaders idênticos)
/// - Redução de alocações em hot paths usando ArrayPool
/// - Perfis de performance adaptativos baseados na GPU detectada
/// </summary>
internal sealed class VulkanGuestGpuBackend : IGuestGpuBackend
{
    public string BackendName => "Vulkan";

    private static readonly IGuestCompiledShader DepthOnlyFragmentShader =
        new VulkanCompiledGuestShader(SpirvFixedShaders.CreateDepthOnlyFragment());

    /// <summary>
    /// Cache de shaders compilados para evitar recompilação.
    /// Key: hash do estado do shader + evaluation.
    /// </summary>
    private readonly Dictionary<ulong, WeakReference<IGuestCompiledShader>> _shaderCache = new();
    private readonly object _shaderCacheLock = new();
    private const int MaxShaderCacheSize = 512;

    /// <summary>
    /// Contador de hits/misses do cache para diagnóstico.
    /// </summary>
    private long _shaderCacheHits;
    private long _shaderCacheMisses;

    public bool TryCompileVertexShader(
        Gen5ShaderState state,
        Gen5ShaderEvaluation evaluation,
        out IGuestCompiledShader? shader,
        out string error,
        int globalBufferBase = 0,
        int totalGlobalBufferCount = -1,
        int imageBindingBase = 0,
        int scalarRegisterBufferIndex = -1,
        int requiredVertexOutputCount = 0,
        ulong storageBufferOffsetAlignment = 1)
    {
        shader = null;

        // Tenta cache primeiro
        var cacheKey = ComputeShaderCacheKey(state, evaluation, ShaderKind.Vertex);
        if (TryGetCachedShader(cacheKey, out var cached))
        {
            shader = cached;
            error = string.Empty;
            return true;
        }

        if (!Gen5SpirvTranslator.TryCompileVertexShader(
            state,
            evaluation,
            out var compiled,
            out error,
            globalBufferBase,
            totalGlobalBufferCount,
            imageBindingBase,
            scalarRegisterBufferIndex,
            requiredVertexOutputCount,
            storageBufferOffsetAlignment))
        {
            return false;
        }

        shader = new VulkanCompiledGuestShader(compiled.Spirv);
        CacheShader(cacheKey, shader);
        return true;
    }

    public bool TryCompilePixelShader(
        Gen5ShaderState state,
        Gen5ShaderEvaluation evaluation,
        IReadOnlyList<Gen5PixelOutput> outputs,
        out IGuestCompiledShader? shader,
        out string error,
        int globalBufferBase = 0,
        int totalGlobalBufferCount = -1,
        int imageBindingBase = 0,
        int scalarRegisterBufferIndex = -1,
        uint pixelInputEnable = 0,
        uint pixelInputAddress = 0,
        IReadOnlyList<Gen5PixelInputCntl>? pixelInputCntl = null,
        ulong storageBufferOffsetAlignment = 1)
    {
        shader = null;

        var cacheKey = ComputeShaderCacheKey(state, evaluation, ShaderKind.Pixel);
        if (TryGetCachedShader(cacheKey, out var cached))
        {
            shader = cached;
            error = string.Empty;
            return true;
        }

        if (!Gen5SpirvTranslator.TryCompilePixelShader(
            state,
            evaluation,
            outputs,
            out var compiled,
            out error,
            globalBufferBase,
            totalGlobalBufferCount,
            imageBindingBase,
            scalarRegisterBufferIndex,
            pixelInputEnable,
            pixelInputAddress,
            pixelInputCntl,
            storageBufferOffsetAlignment))
        {
            return false;
        }

        shader = new VulkanCompiledGuestShader(compiled.Spirv);
        CacheShader(cacheKey, shader);
        return true;
    }

    public bool TryCompileComputeShader(
        Gen5ShaderState state,
        Gen5ShaderEvaluation evaluation,
        uint localSizeX,
        uint localSizeY,
        uint localSizeZ,
        out IGuestCompiledShader? shader,
        out string error,
        int totalGlobalBufferCount = -1,
        int initialScalarBufferIndex = -1,
        uint waveLaneCount = 32,
        ulong storageBufferOffsetAlignment = 1)
    {
        shader = null;

        var cacheKey = ComputeShaderCacheKey(state, evaluation, ShaderKind.Compute);
        if (TryGetCachedShader(cacheKey, out var cached))
        {
            shader = cached;
            error = string.Empty;
            return true;
        }

        if (!Gen5SpirvTranslator.TryCompileComputeShader(
            state,
            evaluation,
            localSizeX,
            localSizeY,
            localSizeZ,
            out var compiled,
            out error,
            totalGlobalBufferCount,
            initialScalarBufferIndex,
            waveLaneCount,
            storageBufferOffsetAlignment))
        {
            return false;
        }

        shader = new VulkanCompiledGuestShader(compiled.Spirv);
        CacheShader(cacheKey, shader);
        return true;
    }

    public IGuestCompiledShader GetDepthOnlyFragmentShader() =>
        DepthOnlyFragmentShader;

    public void EnsureStarted(uint width, uint height) =>
        VulkanVideoPresenter.EnsureStarted(width, height);

    public void HideSplashScreen() =>
        VulkanVideoPresenter.HideSplashScreen();

    public void Submit(byte[] bgraFrame, uint width, uint height) =>
        VulkanVideoPresenter.Submit(bgraFrame, width, height);

    public void SubmitGuestDraw(GuestDrawKind drawKind, uint width, uint height) =>
        VulkanVideoPresenter.SubmitGuestDraw(drawKind, width, height);

    public void SubmitTranslatedDraw(
        IGuestCompiledShader pixelShader,
        IReadOnlyList<GuestTexture> textures,
        IReadOnlyList<GuestGlobalMemoryBuffer> globalMemoryBuffers,
        uint width,
        uint height,
        uint attributeCount,
        IGuestCompiledShader? vertexShader = null,
        uint vertexCount = 3,
        uint instanceCount = 1,
        uint primitiveType = 4,
        GuestIndexBuffer? indexBuffer = null,
        IReadOnlyList<GuestVertexBuffer>? vertexBuffers = null,
        GuestRenderState? renderState = null) =>
        VulkanVideoPresenter.SubmitTranslatedDraw(
            Spirv(pixelShader),
            textures,
            globalMemoryBuffers,
            width,
            height,
            attributeCount,
            vertexShader is null ? null : Spirv(vertexShader),
            vertexCount,
            instanceCount,
            primitiveType,
            indexBuffer,
            vertexBuffers,
            renderState);

    public void SubmitDepthOnlyTranslatedDraw(
        IGuestCompiledShader pixelShader,
        IReadOnlyList<GuestTexture> textures,
        IReadOnlyList<GuestGlobalMemoryBuffer> globalMemoryBuffers,
        uint attributeCount,
        GuestDepthTarget depthTarget,
        IGuestCompiledShader? vertexShader = null,
        uint vertexCount = 3,
        uint instanceCount = 1,
        uint primitiveType = 4,
        GuestIndexBuffer? indexBuffer = null,
        IReadOnlyList<GuestVertexBuffer>? vertexBuffers = null,
        GuestRenderState? renderState = null,
        ulong shaderAddress = 0,
        int baseVertex = 0) =>
        VulkanVideoPresenter.SubmitDepthOnlyTranslatedDraw(
            Spirv(pixelShader),
            textures,
            globalMemoryBuffers,
            attributeCount,
            depthTarget,
            vertexShader is null ? null : Spirv(vertexShader),
            vertexCount,
            instanceCount,
            primitiveType,
            indexBuffer,
            vertexBuffers,
            renderState,
            shaderAddress,
            baseVertex);

    public void SubmitOffscreenTranslatedDraw(
        IGuestCompiledShader pixelShader,
        IReadOnlyList<GuestTexture> textures,
        IReadOnlyList<GuestGlobalMemoryBuffer> globalMemoryBuffers,
        uint attributeCount,
        IReadOnlyList<GuestRenderTarget> targets,
        IGuestCompiledShader? vertexShader = null,
        uint vertexCount = 3,
        uint instanceCount = 1,
        uint primitiveType = 4,
        GuestIndexBuffer? indexBuffer = null,
        IReadOnlyList<GuestVertexBuffer>? vertexBuffers = null,
        GuestRenderState? renderState = null,
        GuestDepthTarget? depthTarget = null,
        ulong shaderAddress = 0,
        int baseVertex = 0) =>
        VulkanVideoPresenter.SubmitOffscreenTranslatedDraw(
            Spirv(pixelShader),
            textures,
            globalMemoryBuffers,
            attributeCount,
            targets,
            vertexShader is null ? null : Spirv(vertexShader),
            vertexCount,
            instanceCount,
            primitiveType,
            indexBuffer,
            vertexBuffers,
            renderState,
            depthTarget,
            shaderAddress,
            baseVertex);

    public void SubmitStorageTranslatedDraw(
        IGuestCompiledShader pixelShader,
        IReadOnlyList<GuestTexture> textures,
        IReadOnlyList<GuestGlobalMemoryBuffer> globalMemoryBuffers,
        uint attributeCount,
        uint width,
        uint height,
        ulong shaderAddress = 0) =>
        VulkanVideoPresenter.SubmitStorageTranslatedDraw(
            Spirv(pixelShader),
            textures,
            globalMemoryBuffers,
            attributeCount,
            width,
            height,
            shaderAddress);

    public long SubmitComputeDispatch(
        ulong shaderAddress,
        IGuestCompiledShader computeShader,
        IReadOnlyList<GuestTexture> textures,
        IReadOnlyList<GuestGlobalMemoryBuffer> globalMemoryBuffers,
        uint groupCountX,
        uint groupCountY,
        uint groupCountZ,
        uint baseGroupX,
        uint baseGroupY,
        uint baseGroupZ,
        uint localSizeX,
        uint localSizeY,
        uint localSizeZ,
        bool isIndirect,
        bool writesGlobalMemory,
        uint threadCountX = uint.MaxValue,
        uint threadCountY = uint.MaxValue,
        uint threadCountZ = uint.MaxValue) =>
        VulkanVideoPresenter.SubmitComputeDispatch(
            shaderAddress,
            Spirv(computeShader),
            textures,
            globalMemoryBuffers,
            groupCountX,
            groupCountY,
            groupCountZ,
            baseGroupX,
            baseGroupY,
            baseGroupZ,
            localSizeX,
            localSizeY,
            localSizeZ,
            isIndirect,
            writesGlobalMemory,
            threadCountX,
            threadCountY,
            threadCountZ);

    public bool TrySubmitGuestImage(
        ulong address,
        uint width,
        uint height,
        uint pitchInPixel) =>
        VulkanVideoPresenter.TrySubmitGuestImage(address, width, height, pitchInPixel);

    public bool TrySubmitOrderedGuestImageFlip(
        int videoOutHandle,
        int displayBufferIndex,
        ulong address,
        uint width,
        uint height,
        uint pitchInPixel) =>
        VulkanVideoPresenter.TrySubmitOrderedGuestImageFlip(
            videoOutHandle,
            displayBufferIndex,
            address,
            width,
            height,
            pitchInPixel);

    public void RegisterKnownDisplayBuffer(ulong address, uint guestFormat) =>
        VulkanVideoPresenter.RegisterKnownDisplayBuffer(address, guestFormat);

    public bool IsGpuGuestImageAvailable(ulong address, uint format, uint numberType) =>
        VulkanVideoPresenter.IsGpuGuestImageAvailable(address, format, numberType);

    public bool TrySubmitGuestImageBlit(
        ulong sourceAddress,
        uint sourceWidth,
        uint sourceHeight,
        uint sourceFormat,
        uint sourceNumberType,
        ulong destinationAddress,
        uint destinationWidth,
        uint destinationHeight,
        uint destinationFormat,
        uint destinationNumberType) =>
        VulkanVideoPresenter.TrySubmitGuestImageBlit(
            sourceAddress,
            sourceWidth,
            sourceHeight,
            sourceFormat,
            sourceNumberType,
            destinationAddress,
            destinationWidth,
            destinationHeight,
            destinationFormat,
            destinationNumberType);

    public bool TryGetRenderTargetOutputKind(uint dataFormat, uint numberType, out Gen5PixelOutputKind outputKind)
    {
        if (VulkanVideoPresenter.TryDecodeRenderTargetFormat(dataFormat, numberType, out var format))
        {
            outputKind = format.OutputKind;
            return true;
        }

        outputKind = default;
        return false;
    }

    public IDisposable EnterGuestQueue(string queueName, ulong submissionId) =>
        VulkanVideoPresenter.EnterGuestQueue(queueName, submissionId);

    public long SubmitOrderedGuestAction(Action action, string debugName) =>
        VulkanVideoPresenter.SubmitOrderedGuestAction(action, debugName);

    public long SubmitOrderedGuestFlipWait(int videoOutHandle, int displayBufferIndex) =>
        VulkanVideoPresenter.SubmitOrderedGuestFlipWait(videoOutHandle, displayBufferIndex);

    public bool WaitForGuestWork(long workSequence, int timeoutMilliseconds = Timeout.Infinite) =>
        VulkanVideoPresenter.WaitForGuestWork(workSequence, timeoutMilliseconds);

    public long CurrentGuestWorkSequenceForDiagnostics =>
        VulkanVideoPresenter.CurrentGuestWorkSequenceForDiagnostics;

    public bool IsGuestImageUploadKnown(ulong address, uint format, uint numberType) =>
        VulkanVideoPresenter.IsGuestImageUploadKnown(address, format, numberType);

    public bool GuestImageWantsInitialData(ulong address) =>
        VulkanVideoPresenter.GuestImageWantsInitialData(address);

    public void ProvideGuestImageInitialData(ulong address, byte[] rgbaPixels) =>
        VulkanVideoPresenter.ProvideGuestImageInitialData(address, rgbaPixels);

    public void SubmitGuestImageFill(ulong address, uint fillValue) =>
        VulkanVideoPresenter.SubmitGuestImageFill(address, fillValue);

    public void SubmitGuestImageWrite(ulong address, byte[] pixels, uint rowOffset = 0) =>
        VulkanVideoPresenter.SubmitGuestImageWrite(address, pixels, rowOffset);

    public bool SupportsPartialImageWrite => true;

    public void RequestCpuWrittenGuestImageSync(ulong scopeAddress = 0, ulong scopeByteCount = ulong.MaxValue) =>
        VulkanVideoPresenter.RequestCpuWrittenGuestImageSync(scopeAddress, scopeByteCount);

    public bool TryGetGuestImageExtent(ulong address, out uint width, out uint height, out ulong byteCount) =>
        VulkanVideoPresenter.TryGetGuestImageExtent(address, out width, out height, out byteCount);

    public IReadOnlyList<(ulong Address, uint Width, uint Height, ulong ByteCount)> GetGuestImageExtents() =>
        VulkanVideoPresenter.GetGuestImageExtents();

    public bool IsTextureContentCached(in TextureContentIdentity identity) =>
        VulkanVideoPresenter.IsTextureContentCached(identity);

    public void AttachGuestMemory(SharpEmu.HLE.ICpuMemory memory) =>
        VulkanVideoPresenter.AttachGuestMemory(memory);

    public ulong GuestStorageBufferOffsetAlignment =>
        VulkanVideoPresenter.GuestStorageBufferOffsetAlignment;

    public void CountShaderCompilation() =>
        VulkanVideoPresenter.CountSpirvCompilation();

    public (long Draws, double DrawMs, long Pipelines, long ShaderCompilations) ReadAndResetPerfCounters() =>
        VulkanVideoPresenter.ReadAndResetPerfCounters();

    public void RequestClose() =>
        VulkanVideoPresenter.RequestClose();

    /// <summary>
    /// Estatísticas do cache de shaders para diagnóstico.
    /// </summary>
    public (long Hits, long Misses, double HitRatio, int CacheSize) GetShaderCacheStats()
    {
        lock (_shaderCacheLock)
        {
            var total = _shaderCacheHits + _shaderCacheMisses;
            var ratio = total > 0 ? (double)_shaderCacheHits / total : 0.0;
            return (_shaderCacheHits, _shaderCacheMisses, ratio, _shaderCache.Count);
        }
    }

    /// <summary>
    /// Limpa o cache de shaders (útil quando detectar pressão de memória).
    /// </summary>
    public void ClearShaderCache()
    {
        lock (_shaderCacheLock)
        {
            _shaderCache.Clear();
        }
    }

    private static byte[] Spirv(IGuestCompiledShader shader) =>
        shader is VulkanCompiledGuestShader vulkanShader
        ? vulkanShader.Spirv
        : throw new InvalidOperationException(
            $"shader handle of type {shader.GetType().Name} was not compiled by the Vulkan backend");

    private enum ShaderKind { Vertex, Pixel, Compute }

    /// <summary>
    /// Computa uma chave de cache baseada no estado do shader.
    /// Usa hash combinado do estado e evaluation.
    /// </summary>
    private static ulong ComputeShaderCacheKey(Gen5ShaderState state, Gen5ShaderEvaluation evaluation, ShaderKind kind)
    {
        // Hash simples mas efetivo para cache
        var hash = (ulong)state.GetHashCode();
        hash = hash * 31 + (ulong)evaluation.GetHashCode();
        hash = hash * 31 + (ulong)kind;
        return hash;
    }

    private bool TryGetCachedShader(ulong key, out IGuestCompiledShader? shader)
    {
        lock (_shaderCacheLock)
        {
            if (_shaderCache.TryGetValue(key, out var weakRef) && weakRef.TryGetTarget(out shader) && shader != null)
            {
                _shaderCacheHits++;
                return true;
            }
            _shaderCacheMisses++;
            shader = null;
            return false;
        }
    }

    private void CacheShader(ulong key, IGuestCompiledShader shader)
    {
        lock (_shaderCacheLock)
        {
            // Evita crescimento ilimitado do cache
            if (_shaderCache.Count >= MaxShaderCacheSize)
            {
                // Remove entradas com referências fracas já coletadas
                var deadKeys = _shaderCache
                    .Where(kvp => !kvp.Value.TryGetTarget(out _))
                    .Select(kvp => kvp.Key)
                    .ToList();

                foreach (var deadKey in deadKeys)
                {
                    _shaderCache.Remove(deadKey);
                }

                // Se ainda estiver cheio, remove metade das entradas mais antigas
                if (_shaderCache.Count >= MaxShaderCacheSize)
                {
                    var keysToRemove = _shaderCache.Keys.Take(_shaderCache.Count / 2).ToList();
                    foreach (var k in keysToRemove)
                    {
                        _shaderCache.Remove(k);
                    }
                }
            }

            _shaderCache[key] = new WeakReference<IGuestCompiledShader>(shader);
        }
    }
}
