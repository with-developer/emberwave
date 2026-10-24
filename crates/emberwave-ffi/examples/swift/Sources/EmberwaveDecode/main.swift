// Decodes a PWV1 file with the emberwave C API and writes the frames to a Y4M
// file, as decode.c does in C: the C API as Swift sees it through the module
// Emberwave of the XCFramework.
//
//     emberwave-decode INPUT.pwv1 OUTPUT.y4m

import Emberwave
import Foundation

/// A failure, described for the user.
struct Failure: Error, CustomStringConvertible {
    let description: String
}

/// Throws unless `status` is `EMBERWAVE_STATUS_OK`.
func check(_ status: EmberwaveStatus, _ call: String) throws {
    if status != EMBERWAVE_STATUS_OK {
        let why = String(cString: emberwave_last_error())
        throw Failure(description: "\(call) failed (status \(status.rawValue)): \(why)")
    }
}

/// Writes all of `bytes` to `stream`.
func write(_ bytes: [UInt8], to stream: OutputStream, path: String) throws {
    var written = 0
    while written < bytes.count {
        let count = bytes[written...].withUnsafeBufferPointer {
            stream.write($0.baseAddress!, maxLength: $0.count)
        }
        if count <= 0 {
            let why = stream.streamError?.localizedDescription ?? "nothing written"
            throw Failure(description: "\(path): \(why)")
        }
        written += count
    }
}

func decode(input: String, output: String) throws {
    guard let file = FileManager.default.contents(atPath: input) else {
        throw Failure(description: "\(input): cannot read it")
    }
    let bytes = [UInt8](file)

    // The file header says how large the pictures are.
    var stream = EmberwaveStreamInfo()
    try check(emberwave_probe_pwv1(bytes, bytes.count, &stream), "emberwave_probe_pwv1")

    // One context per GPU; its decoders share the compiled shaders.
    var context: OpaquePointer?
    defer { emberwave_context_destroy(context) }
    try check(emberwave_context_create(nil, &context), "emberwave_context_create")
    let version = String(cString: emberwave_version())
    print("emberwave \(version) on \(String(cString: emberwave_context_adapter(context)))")

    var decoder: OpaquePointer?
    defer { emberwave_decoder_destroy(decoder) }
    try check(
        emberwave_decoder_create(context, stream.width, stream.height, stream.chroma, &decoder),
        "emberwave_decoder_create")
    var frame = EmberwaveFrameInfo()
    try check(emberwave_decoder_info(decoder, &frame), "emberwave_decoder_info")

    // The three planes, back to back, as a Y4M frame holds them.
    let lumaBytes = Int(frame.width) * Int(frame.height)
    let chromaBytes = Int(frame.chroma_width) * Int(frame.chroma_height)
    var planes = [UInt8](repeating: 0, count: lumaBytes + 2 * chromaBytes)

    guard let out = OutputStream(toFileAtPath: output, append: false) else {
        throw Failure(description: "\(output): cannot write it")
    }
    out.open()
    defer { out.close() }
    if let error = out.streamError {
        throw Failure(description: "\(output): \(error.localizedDescription)")
    }

    let chromaTag: String
    if frame.chroma == EMBERWAVE_CHROMA_444 {
        chromaTag = "C444"
    } else if stream.chroma_siting == EMBERWAVE_SITING_LEFT {
        chromaTag = "C420mpeg2"
    } else {
        chromaTag = "C420jpeg"
    }
    let rate =
        stream.frame_rate_num != 0 && stream.frame_rate_den != 0
        ? "\(stream.frame_rate_num):\(stream.frame_rate_den)" : "30:1"
    let range = stream.range == EMBERWAVE_RANGE_FULL ? "FULL" : "LIMITED"
    let header =
        "YUV4MPEG2 W\(frame.width) H\(frame.height) F\(rate) Ip A1:1 XCOLORRANGE=\(range) \(chromaTag)\n"
    try write(Array(header.utf8), to: out, path: output)

    var offset = 24  // past the file header
    var frames = 0
    while offset + 4 <= bytes.count {
        let size =
            Int(bytes[offset]) | Int(bytes[offset + 1]) << 8 | Int(bytes[offset + 2]) << 16
            | Int(bytes[offset + 3]) << 24
        offset += 4
        guard size <= bytes.count - offset else {
            throw Failure(description: "frame \(frames) is cut off")
        }

        // A whole frame of the file in one call; a network client would push
        // each packet as it arrives.
        try bytes.withUnsafeBufferPointer {
            try check(
                emberwave_decoder_push_packet(decoder, $0.baseAddress! + offset, size),
                "emberwave_decoder_push_packet")
        }
        offset += size
        guard emberwave_decoder_is_ready(decoder, false) else {
            throw Failure(description: "frame \(frames) is incomplete")
        }

        try planes.withUnsafeMutableBufferPointer {
            let base = $0.baseAddress!
            let y = EmberwavePlane(data: base, stride: Int(frame.width))
            let cb = EmberwavePlane(data: base + lumaBytes, stride: Int(frame.chroma_width))
            let cr = EmberwavePlane(
                data: base + lumaBytes + chromaBytes, stride: Int(frame.chroma_width))
            try check(
                emberwave_decoder_decode_yuv(decoder, y, cb, cr), "emberwave_decoder_decode_yuv")
        }
        try write(Array("FRAME\n".utf8), to: out, path: output)
        try write(planes, to: out, path: output)
        frames += 1
    }
    guard offset == bytes.count else {
        throw Failure(description: "the file ends inside the size of frame \(frames)")
    }
    print("decoded \(frames) frames of \(frame.width)x\(frame.height)")
}

let arguments = CommandLine.arguments
guard arguments.count == 3 else {
    FileHandle.standardError.write(Data("usage: \(arguments[0]) INPUT.pwv1 OUTPUT.y4m\n".utf8))
    exit(2)
}
do {
    try decode(input: arguments[1], output: arguments[2])
} catch {
    FileHandle.standardError.write(Data("\(error)\n".utf8))
    exit(1)
}
