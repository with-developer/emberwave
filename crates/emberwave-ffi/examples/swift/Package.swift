// swift-tools-version:5.9

// The Swift example of the C API: emberwave-decode, decode.c in Swift. It
// links the XCFramework that scripts/build_xcframework.sh makes, from the
// root of the repository:
//
//     scripts/build_xcframework.sh macos-arm64
//     swift run --package-path crates/emberwave-ffi/examples/swift \
//         emberwave-decode testdata/gradient_200x120_420.pwv1 out.y4m
//
// An app that uses a release's XCFramework names it in the same way, by its
// path or as a remote binary target with its URL and checksum.

import PackageDescription

let package = Package(
    name: "EmberwaveExample",
    platforms: [.macOS(.v11), .iOS(.v12)],
    targets: [
        .binaryTarget(
            name: "Emberwave",
            path: "../../../../target/xcframework/emberwave.xcframework"
        ),
        .executableTarget(
            name: "emberwave-decode",
            dependencies: ["Emberwave"],
            path: "Sources/EmberwaveDecode",
            // The system libraries of native-static-libs-*.txt beside the
            // XCFramework; the Swift runtime brings the rest.
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("QuartzCore"),
                .linkedFramework("CoreGraphics"),
                .linkedFramework("Foundation"),
                .linkedLibrary("iconv"),
                .linkedLibrary("objc"),
            ]
        ),
    ]
)
