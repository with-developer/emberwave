// The test pictures: the synthetic scene of the browser benchmark
// (crates/emberwave-web/src/bench.rs), so that the phone's numbers compare
// with the browser's. A gradient, colour bars, a shaded disc that moves from
// picture to picture, a fine grid and a little grain, which the encoder finds
// about as hard as a game's picture.

import Foundation

/// Picture `index` of the scene in 4:2:0: the Y, Cb and Cr planes, rows packed.
func scene(width: Int, height: Int, index: Int) -> [[UInt8]] {
    let (w, h) = (Float(width), Float(height))
    var grain = UInt32(0x9e37_79b9) ^ (UInt32(truncatingIfNeeded: index) &* 0x85eb_ca6b)
    func noise() -> Float {
        grain ^= grain << 13
        grain ^= grain >> 17
        grain ^= grain << 5
        return Float(grain >> 29) - 3.5
    }
    let (cx, cy) = (w * (0.3 + 0.05 * Float(index)), h * (0.5 + 0.03 * Float(index)))
    let radius = h * 0.25
    let bars: [Float] = [235, 210, 170, 145, 106, 81, 41, 16]

    var luma = [UInt8](repeating: 0, count: width * height)
    for row in 0..<height {
        let y = Float(row)
        for column in 0..<width {
            let x = Float(column)
            var value: Float =
                y < h * 0.2
                ? bars[column * 8 / width] : 40 + 160 * x / w * (0.6 + 0.4 * y / h)
            let (dx, dy) = (x - cx, y - cy)
            let distance = (dx * dx + dy * dy).squareRoot()
            if distance < radius {
                value = 60 + 150 * (1 - distance / radius) + 20 * (dy / radius)
            }
            if column % 16 == 0 || row % 16 == 0 {
                value = value * 0.8 + 30
            }
            luma[row * width + column] = UInt8(min(max(value + noise(), 0), 255))
        }
    }

    let (cw, ch) = (width / 2, height / 2)
    func chroma(_ phase: Float) -> [UInt8] {
        var plane = [UInt8](repeating: 0, count: cw * ch)
        for row in 0..<ch {
            let y = Float(row) / Float(ch)
            for column in 0..<cw {
                let x = Float(column) / Float(cw)
                plane[row * cw + column] = UInt8(
                    128 + 60 * (sin(x * 6 + phase) * cos(y * 4 - phase)))
            }
        }
        return plane
    }
    return [luma, chroma(Float(index) * 0.4), chroma(1.7 + Float(index) * 0.3)]
}
