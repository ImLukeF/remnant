import Foundation

/// Eight short, DC-removed spectra per second. No waveform is retained beyond one second.
struct SpectralAccumulator: Sendable {
    let sampleRate: Int
    private let size: Int
    private let window: [Double]
    private let rootsReal: [Double]
    private let rootsImaginary: [Double]
    private let filters: [[(Int, Double)]]
    private var samples: [Float] = []
    private var real: [Double]
    private var imaginary: [Double]
    private var count = 0
    private var means = Array(repeating: 0.0, count: 24)
    private var moments = Array(repeating: 0.0, count: 24)
    private var entropyMean = 0.0
    private var powerMean = 0.0
    private var previousBands: [Double]?
    private var fluxCount = 0
    private var fluxMean = 0.0
    private var fluxMoment = 0.0
    private let retainFrames: Bool
    private var acousticFrames: [Float] = []

    init(sampleRate: Int, retainFrames: Bool = false) {
        self.sampleRate = sampleRate
        self.retainFrames = retainFrames
        if retainFrames { acousticFrames.reserveCapacity(240*25) }
        var size = 1
        while size * 2 <= sampleRate / 8 { size *= 2 }
        self.size = size
        self.window = (0..<size).map { 0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(size - 1)) }
        self.rootsReal = (0..<(size / 2)).map { cos(-2 * Double.pi * Double($0) / Double(size)) }
        self.rootsImaginary = (0..<(size / 2)).map { sin(-2 * Double.pi * Double($0) / Double(size)) }
        let low = 2595 * log10(1 + 80.0 / 700)
        let high = 2595 * log10(1 + 4000.0 / 700)
        let edges = (0..<26).map { 700 * (pow(10, (low + (high - low) * Double($0) / 25) / 2595) - 1) }
        self.filters = (0..<24).map { band in
            (0...(size / 2)).compactMap { index in
                let hz = Double(index * sampleRate) / Double(size)
                let rising = (hz - edges[band]) / (edges[band + 1] - edges[band])
                let falling = (edges[band + 2] - hz) / (edges[band + 2] - edges[band + 1])
                let weight = max(0, min(rising, falling))
                return weight > 0 ? (index, weight) : nil
            }
        }
        self.real = Array(repeating: 0, count: size)
        self.imaginary = Array(repeating: 0, count: size)
        self.samples.reserveCapacity(sampleRate)
    }

    mutating func append(_ value: Float) {
        samples.append(value)
        guard samples.count == sampleRate else { return }
        for frame in 0..<8 {
            let offset = frame * sampleRate / 8
            let average = samples[offset..<(offset + size)].reduce(0.0) { $0 + Double($1) } / Double(size)
            for index in 0..<size {
                real[index] = (Double(samples[offset + index]) - average) * window[index]
                imaginary[index] = 0
            }
            fft()
            let bands = filters.map { filter in
                filter.reduce(0.0) { total, entry in
                    let (index, weight) = entry
                    return total + (real[index] * real[index] + imaginary[index] * imaginary[index]) * weight
                }
            }
            let total = bands.reduce(0, +)
            let normalized = bands.map { $0 / max(total, 1e-300) }
            let relative = normalized.map { 10 * log10(max($0, 1e-12)) }
            if retainFrames {
                acousticFrames += relative.map { Float(($0*1e4).rounded(.toNearestOrEven)/1e4) }
                let level = 10*log10(max(total/Double(size*size), 1e-12))
                acousticFrames.append(Float((level*1e4).rounded(.toNearestOrEven)/1e4))
            }
            count += 1
            for index in 0..<24 {
                let delta = relative[index] - means[index]
                means[index] += delta / Double(count)
                moments[index] += delta * (relative[index] - means[index])
            }
            let entropy = -normalized.reduce(0) { $0 + $1 * log(max($1, 1e-300)) } / log(24.0)
            entropyMean += (entropy - entropyMean) / Double(count)
            powerMean += (total / Double(size * size) - powerMean) / Double(count)
            if let previousBands {
                let flux = zip(normalized, previousBands).reduce(0) { $0 + abs($1.0 - $1.1) }
                fluxCount += 1
                let delta = flux - fluxMean
                fluxMean += delta / Double(fluxCount)
                fluxMoment += delta * (flux - fluxMean)
            }
            previousBands = normalized
        }
        samples.removeAll(keepingCapacity: true)
    }

    mutating func finishEpoch() -> (features: [Double], usable: Bool, frames: [Float]?) {
        let std = moments.map { sqrt(max(0, $0 / Double(max(count, 1)))) }
        let values = means + std + [entropyMean, fluxMean, sqrt(max(0, fluxMoment / Double(max(fluxCount, 1))))]
        let usable = count == 240 && powerMean > 1e-12 && values.allSatisfy(\.isFinite)
        let frames = retainFrames ? acousticFrames : nil
        acousticFrames.removeAll(keepingCapacity: true)
        count = 0
        means = Array(repeating: 0, count: 24)
        moments = Array(repeating: 0, count: 24)
        entropyMean = 0
        powerMean = 0
        previousBands = nil
        fluxCount = 0
        fluxMean = 0
        fluxMoment = 0
        return (values, usable, frames)
    }

    private mutating func fft() {
        var reverse = 0
        for index in 1..<size {
            var bit = size >> 1
            while reverse & bit != 0 { reverse ^= bit; bit >>= 1 }
            reverse ^= bit
            if index < reverse {
                real.swapAt(index, reverse)
                imaginary.swapAt(index, reverse)
            }
        }
        var length = 2
        while length <= size {
            let half = length / 2
            let step = size / length
            for start in stride(from: 0, to: size, by: length) {
                for offset in 0..<half {
                    let a = start + offset, b = a + half
                    let root = offset * step
                    let vr = real[b] * rootsReal[root] - imaginary[b] * rootsImaginary[root]
                    let vi = real[b] * rootsImaginary[root] + imaginary[b] * rootsReal[root]
                    let ur = real[a], ui = imaginary[a]
                    real[a] = ur + vr; imaginary[a] = ui + vi
                    real[b] = ur - vr; imaginary[b] = ui - vi
                }
            }
            length *= 2
        }
    }
}
