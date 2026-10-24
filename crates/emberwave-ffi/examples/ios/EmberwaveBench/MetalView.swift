// Shows a texture emberwave decoded into, drawn with Metal on the device of
// emberwave's context, scaled to fit: what a player does with the texture of
// emberwave_texture_metal.

import Metal
import MetalKit
import SwiftUI

struct MetalView: UIViewRepresentable {
    let device: MTLDevice
    let picture: MTLTexture?

    func makeCoordinator() -> Renderer {
        Renderer(device: device)
    }

    func makeUIView(context: Context) -> MTKView {
        let view = MTKView(frame: .zero, device: device)
        view.colorPixelFormat = .bgra8Unorm
        view.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        // Drawn when the picture changes, not every display refresh.
        view.isPaused = true
        view.enableSetNeedsDisplay = true
        view.delegate = context.coordinator
        return view
    }

    func updateUIView(_ view: MTKView, context: Context) {
        context.coordinator.picture = picture
        view.setNeedsDisplay()
    }

    /// Draws the picture as a quad that keeps its aspect ratio.
    final class Renderer: NSObject, MTKViewDelegate {
        private static let shaders = """
            #include <metal_stdlib>
            using namespace metal;

            struct Corner {
                float4 position [[position]];
                float2 uv;
            };

            vertex Corner quad(uint index [[vertex_id]], constant float2 &scale [[buffer(0)]]) {
                float2 uv = float2(index & 1, index >> 1);
                Corner corner;
                corner.position = float4((uv * 2 - 1) * float2(1, -1) * scale, 0, 1);
                corner.uv = uv;
                return corner;
            }

            fragment float4 picture(Corner in [[stage_in]], texture2d<float> picture [[texture(0)]]) {
                constexpr sampler linear(filter::linear);
                return picture.sample(linear, in.uv);
            }
            """

        var picture: MTLTexture?
        private let queue: MTLCommandQueue?
        private let pipeline: MTLRenderPipelineState?

        init(device: MTLDevice) {
            queue = device.makeCommandQueue()
            let descriptor = MTLRenderPipelineDescriptor()
            if let library = try? device.makeLibrary(source: Self.shaders, options: nil) {
                descriptor.vertexFunction = library.makeFunction(name: "quad")
                descriptor.fragmentFunction = library.makeFunction(name: "picture")
            }
            descriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            pipeline = try? device.makeRenderPipelineState(descriptor: descriptor)
        }

        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
            view.setNeedsDisplay()
        }

        func draw(in view: MTKView) {
            guard let pass = view.currentRenderPassDescriptor, let drawable = view.currentDrawable,
                let commands = queue?.makeCommandBuffer(),
                let encoder = commands.makeRenderCommandEncoder(descriptor: pass)
            else {
                return
            }
            if let picture, let pipeline {
                let viewAspect = Float(view.drawableSize.width / max(view.drawableSize.height, 1))
                let pictureAspect = Float(picture.width) / Float(picture.height)
                var scale =
                    pictureAspect > viewAspect
                    ? SIMD2<Float>(1, viewAspect / pictureAspect)
                    : SIMD2<Float>(pictureAspect / viewAspect, 1)
                encoder.setRenderPipelineState(pipeline)
                encoder.setVertexBytes(&scale, length: MemoryLayout<SIMD2<Float>>.size, index: 0)
                encoder.setFragmentTexture(picture, index: 0)
                encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            }
            encoder.endEncoding()
            commands.present(drawable)
            commands.commit()
        }
    }
}
