import Foundation

/// A second reading of a finished night. It hears the whole recording first, so every sound is measured against that
/// night's own quiet level and each half-minute is read with the 20 minutes either side of it.
public struct MorningModel: Decodable, Sendable {
    public let identifier: String
    let coefficients: [[Double]]
    let intercepts: [Double]

    public static let shipped = try! JSONDecoder().decode(MorningModel.self, from: Data(MorningModelData.json.utf8))
    /// Shorter recordings have too little night to set a quiet level from.
    public static let minimumEpochs = 60

    /// The chance of awake, light, deep and REM for each half-minute, from `MorningSound.rows`.
    public func probabilities(_ rows: [[Double]]) -> [[Double]] {
        Self.context(rows).map { features in
            let scores = zip(coefficients, intercepts).map { weights, intercept in
                zip(weights, features).reduce(intercept) { $0 + $1.0 * $1.1 }
            }
            let top = scores.max() ?? 0
            let raised = scores.map { exp($0 - top) }
            let total = raised.reduce(0, +)
            return raised.map { $0 / total }
        }
    }

    public func stages(_ rows: [[Double]]) -> [SleepStage] {
        probabilities(rows).map { chances in SleepStage.allCases[chances.indices.max { chances[$0] < chances[$1] } ?? 1] }
    }

    /// Each measurement scaled to the night, then its average and spread over 5 and 20 minutes either side.
    static func context(_ rows: [[Double]]) -> [[Double]] {
        guard let width = rows.first?.count else { return [] }
        var scaled = rows
        for column in 0..<width {
            let values = rows.map { $0[column] }
            let middle = median(values), spread = deviation(values[...]) + 1e-6
            for row in rows.indices { scaled[row][column] = (rows[row][column] - middle) / spread }
        }
        let core = 48..<width  // loudness, movement and breathing
        return rows.indices.map { row in
            let near = scaled[max(0, row - 5)...min(rows.count - 1, row + 5)]
            let far = scaled[max(0, row - 20)...min(rows.count - 1, row + 20)]
            var features = scaled[row]
            for window in [near, far] {
                features += (0..<width).map { column in window.reduce(0) { $0 + $1[column] } / Double(window.count) }
            }
            for window in [near, far] {
                features += core.map { column in deviation(window.map { $0[column] }[...]) }
            }
            return features
        }
    }

    static func median(_ values: [Double]) -> Double {
        let sorted = values.sorted(), half = sorted.count / 2
        return sorted.count % 2 == 1 ? sorted[half] : (sorted[half - 1] + sorted[half]) / 2
    }

    static func deviation(_ values: ArraySlice<Double>) -> Double {
        let mean = values.reduce(0, +) / Double(values.count)
        return (values.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(values.count)).squareRoot()
    }
}

/// Turns a night's sound into one row of measurements per half-minute. Feed it the 16 kHz recording in order.
public struct MorningSound: Sendable {
    public static let sampleRate = 16_000
    static let size = 1_024, hop = 512, bands = 24
    static let framesPerSecond = Double(sampleRate) / Double(hop)
    static let epochFrames = Int(30 * framesPerSecond)

    private let window: [Double]
    private let filters: [[(Int, Double)]]
    private let reversed: [Int]
    private let rootsReal: [Double]
    private let rootsImaginary: [Double]
    private var real = [Double](repeating: 0, count: size)
    private var imaginary = [Double](repeating: 0, count: size)
    private var pending: [Float] = []
    private var levels: [Float] = []      // 24 band levels in dB for every frame
    private var centroids: [Double] = []
    private var flatness: [Double] = []

    public init() {
        let size = Self.size
        window = (0..<size).map { 0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(size - 1)) }
        let mel = { (hz: Double) in 2595 * log10(1 + hz / 700) }
        let low = mel(60), high = mel(8_000)
        let edges = (0..<(Self.bands + 2)).map { 700 * (pow(10, (low + (high - low) * Double($0) / Double(Self.bands + 1)) / 2595) - 1) }
        filters = (0..<Self.bands).map { band in
            (0...(size / 2)).compactMap { index in
                let hz = Double(index * Self.sampleRate) / Double(size)
                let weight = min((hz - edges[band]) / (edges[band + 1] - edges[band]), (edges[band + 2] - hz) / (edges[band + 2] - edges[band + 1]))
                return weight > 0 ? (index, weight) : nil
            }
        }
        let bits = size.trailingZeroBitCount
        reversed = (0..<size).map { index in (0..<bits).reduce(0) { $0 | ((index >> $1) & 1) << (bits - 1 - $1) } }
        rootsReal = (0..<(size / 2)).map { cos(-2 * Double.pi * Double($0) / Double(size)) }
        rootsImaginary = (0..<(size / 2)).map { sin(-2 * Double.pi * Double($0) / Double(size)) }
    }

    public mutating func append(_ samples: UnsafeBufferPointer<Float>) {
        pending.append(contentsOf: samples)
        var offset = 0
        while pending.count - offset >= Self.size { frame(at: offset); offset += Self.hop }
        pending.removeFirst(offset)
    }

    private mutating func frame(at offset: Int) {
        let size = Self.size
        var mean = 0.0
        for index in 0..<size { mean += Double(pending[offset + index]) }
        mean /= Double(size)
        for index in 0..<size {
            real[reversed[index]] = (Double(pending[offset + index]) - mean) * window[index]
            imaginary[index] = 0
        }
        var width = 2
        while width <= size {
            let half = width / 2, step = size / width
            var start = 0
            while start < size {
                for index in 0..<half {
                    let rootReal = rootsReal[index * step], rootImaginary = rootsImaginary[index * step]
                    let a = start + index, b = a + half
                    let oddReal = real[b] * rootReal - imaginary[b] * rootImaginary
                    let oddImaginary = real[b] * rootImaginary + imaginary[b] * rootReal
                    real[b] = real[a] - oddReal; imaginary[b] = imaginary[a] - oddImaginary
                    real[a] += oddReal; imaginary[a] += oddImaginary
                }
                start += width
            }
            width *= 2
        }
        for index in 0...(size / 2) { real[index] = real[index] * real[index] + imaginary[index] * imaginary[index] + 1e-20 }
        for filter in filters {
            levels.append(Float(10 * log10(filter.reduce(0) { $0 + real[$1.0] * $1.1 } + 1e-20)))
        }
        var total = 0.0, weighted = 0.0, logs = 0.0
        for index in 4...(size / 2) {
            total += real[index]; logs += log(real[index])
            weighted += real[index] * Double(index * Self.sampleRate) / Double(size)
        }
        let count = Double(size / 2 - 3)
        centroids.append(weighted / total)
        flatness.append(exp(logs / count) / (total / count))
    }

    /// One row for each half-minute that starts `starts` seconds into the recording; nil where the sound doesn't cover it.
    public func rows(starts: [TimeInterval]) -> [[Double]?] {
        let bands = Self.bands, frames = centroids.count, per = Self.epochFrames
        guard frames >= per else { return starts.map { _ in nil } }
        let floors = (0..<bands).map { band in
            MorningModel.median((0..<frames).map { Double(levels[$0 * bands + band]) })
        }
        let loud = (0..<frames).map { frame in
            10 * log10((4..<16).reduce(0) { $0 + pow(10, (Double(levels[frame * bands + $1]) - floors[$1]) / 10) } / 12)
        }
        return starts.map { seconds in
            let first = Int(seconds * Self.framesPerSecond)
            guard seconds >= 0, first <= frames - per else { return nil }
            let range = first..<(first + per)
            var means = [Double](repeating: 0, count: bands), spreads = [Double](repeating: 0, count: bands)
            for band in 0..<bands {
                let values = range.map { Double(levels[$0 * bands + band]) - floors[band] }
                means[band] = values.reduce(0, +) / Double(per)
                spreads[band] = MorningModel.deviation(values[...])
            }
            let level = loud[range], sorted = level.sorted()
            let percentile = { (share: Double) -> Double in
                let position = share * Double(per - 1), below = Int(position)
                return sorted[below] + (sorted[min(below + 1, per - 1)] - sorted[below]) * (position - Double(below))
            }
            let jitter = zip(level.dropFirst(), level).reduce(0) { $0 + abs($1.0 - $1.1) } / Double(per - 1)
            let wide = Self.breathing(loud[max(0, first - per)..<min(frames, first + 2 * per)])
            let narrow = Self.breathing(level)
            let pitch = centroids[range]
            return means + spreads + [
                level.reduce(0, +) / Double(per), MorningModel.deviation(level), percentile(0.95) - percentile(0.05),
                Double(level.count { $0 > 10 }) / Double(per), Double(level.count { $0 > 20 }) / Double(per), jitter,
                pitch.reduce(0, +) / Double(per), MorningModel.deviation(pitch),
                flatness[range].reduce(0) { $0 + log($1) } / Double(per),
                wide.strength, wide.period, narrow.strength, narrow.period,
            ]
        }
    }

    /// How strongly, and how often, the loudness repeats every 2 to 8 seconds.
    static func breathing(_ envelope: ArraySlice<Double>) -> (strength: Double, period: Double) {
        let step = 8, rate = framesPerSecond / Double(step)
        let values = Array(envelope)
        let coarse = (0..<(values.count / step)).map { block in values[(block * step)..<((block + 1) * step)].reduce(0, +) / Double(step) }
        let kernel = Int(rate * 10) | 1, half = kernel / 2
        guard coarse.count > Int(8 * rate) else { return (0, 0) }
        let steady = coarse.indices.map { index -> Double in
            let around = ((index - half)...(index + half)).reduce(0.0) { $0 + coarse[min(max($1, 0), coarse.count - 1)] } / Double(kernel)
            return min(6, max(-6, coarse[index] - around))
        }
        guard MorningModel.deviation(steady[...]) >= 1e-6 else { return (0, 0) }
        let match = { (lag: Int) in (0..<(steady.count - lag)).reduce(0.0) { $0 + steady[$1] * steady[$1 + lag] } }
        let whole = match(0)
        var best = Int(2 * rate), strength = -Double.infinity
        for lag in Int(2 * rate)..<Int(8 * rate) {
            let value = match(lag) / whole
            if value > strength { strength = value; best = lag }
        }
        return (strength, Double(best) / rate)
    }
}
