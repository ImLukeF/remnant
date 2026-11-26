// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "SleepWakeKit",
    platforms: [.iOS(.v18), .macOS(.v15)],
    products: [
        .library(name: "SleepWakeKit", targets: ["SleepWakeKit"]),
        .library(name: "SleepWakeAudio", targets: ["SleepWakeAudio"]),
    ],
    targets: [
        .target(name: "SleepWakeKit"),
        .target(name: "SleepWakeAudio", dependencies: ["SleepWakeKit"]),
        .testTarget(name: "SleepWakeKitTests", dependencies: ["SleepWakeKit", "SleepWakeAudio"], resources: [.copy("spectral-reference.json"), .copy("gru-reference.json"), .copy("acoustic-reference.json"), .copy("acoustic-model-reference.json"), .copy("morning-reference.json")]),
    ]
)
