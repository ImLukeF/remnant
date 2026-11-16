import Foundation

/// An observation of audio, not a sleep-stage diagnosis. No waveform leaves the extractor.
public struct AudioEpoch: Codable, Equatable, Sendable {
    public let start: Date
    public let end: Date
    public let features: [Double]
    public let usable: Bool

    public init(start: Date, end: Date, features: [Double], usable: Bool) {
        self.start = start
        self.end = end
        self.features = features
        self.usable = usable
    }
}

/// Streaming, bounded-memory features shared with the reference training implementation.
/// The sample rate must be an integral number of samples per second.
public struct AudioFeatureExtractor: Sendable {
    public static let featureNames = [
        "mean_dbfs", "std_dbfs", "min_dbfs", "max_dbfs",
        "mean_zero_crossing_rate", "clipped_fraction", "envelope_autocorrelation",
    ]
    public static let epochSeconds = 30
    public static let spectralFeatureNames = featureNames
        + ["mean", "std"].flatMap { kind in (0..<24).map { "mel_relative_\(kind)_" + ($0 < 10 ? "0" : "") + String($0) } }
        + ["spectral_entropy_mean", "spectral_flux_mean", "spectral_flux_std"]
    public let sampleRate: Int
    private var start: Date
    private var samples = 0
    private var squares = 0.0
    private var crossings = 0
    private var clipped = 0
    private var previous: Float?
    private var levels: [Double] = []
    private var crossingRates: [Double] = []
    private var clippedFractions: [Double] = []
    private var invalid = false
    private var spectrum: SpectralAccumulator?

    public init(sampleRate: Int, start: Date, spectral: Bool = false) throws {
        guard (8_000...192_000).contains(sampleRate) else { throw FeatureError.invalidSampleRate }
        self.sampleRate = sampleRate
        self.start = start
        self.spectrum = spectral ? SpectralAccumulator(sampleRate: sampleRate) : nil
    }

    public mutating func append(_ pcm: [Float]) -> [AudioEpoch] {
        pcm.withUnsafeBufferPointer { append($0) }
    }

    public mutating func append(_ pcm: UnsafeBufferPointer<Float>) -> [AudioEpoch] {
        var epochs: [AudioEpoch] = []
        for raw in pcm {
            let value = raw.isFinite ? raw : 0
            invalid = invalid || !raw.isFinite || abs(value) > 1
            spectrum?.append(value)
            squares += Double(value) * Double(value)
            if abs(value) >= 0.999 { clipped += 1 }
            if let previous, (previous < 0) != (value < 0) { crossings += 1 }
            previous = value
            samples += 1
            guard samples == sampleRate else { continue }
            levels.append(10 * log10(max(squares / Double(samples), 1e-12)))
            crossingRates.append(Double(crossings) / Double(samples))
            clippedFractions.append(Double(clipped) / Double(samples))
            samples = 0
            squares = 0
            crossings = 0
            clipped = 0
            if levels.count == Self.epochSeconds {
                let mean = levels.reduce(0, +) / Double(levels.count)
                let variance = levels.map { pow($0 - mean, 2) }.reduce(0, +) / Double(levels.count)
                let clip = clippedFractions.reduce(0, +) / Double(levels.count)
                var values = [
                    mean, sqrt(variance), levels.min()!, levels.max()!,
                    crossingRates.reduce(0, +) / Double(levels.count), clip,
                    Self.periodicity(levels),
                ]
                let spectral = spectrum?.finishEpoch()
                if let spectral { values += spectral.features }
                let end = start.addingTimeInterval(Double(Self.epochSeconds))
                epochs.append(AudioEpoch(start: start, end: end, features: values,
                                         usable: !invalid && mean > -119 && clip < 0.01 && (spectral?.usable ?? true)))
                start = end
                levels.removeAll(keepingCapacity: true)
                crossingRates.removeAll(keepingCapacity: true)
                clippedFractions.removeAll(keepingCapacity: true)
                invalid = false
                previous = nil
            }
        }
        return epochs
    }

    private static func periodicity(_ values: [Double]) -> Double {
        let mean = values.reduce(0, +) / Double(values.count)
        let centered = values.map { $0 - mean }
        let energy = centered.map { $0 * $0 }.reduce(0, +)
        guard energy > 1e-9 else { return 0 }
        return (2...10).map { lag in
            zip(centered.dropFirst(lag), centered).map(*).reduce(0, +) / energy
        }.max() ?? 0
    }

    public enum FeatureError: Error { case invalidSampleRate }
}
