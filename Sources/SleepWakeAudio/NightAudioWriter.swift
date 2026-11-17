#if canImport(AVFAudio)
import AVFAudio
import Foundation

/// Confined to the capture worker after construction. Lossless mono PCM, with a bounded resampling buffer.
final class NightAudioWriter: @unchecked Sendable {
    enum Failure: Error { case invalidFormat, conversion, tooLong }
    private let source: AVAudioFormat
    private let converter: AVAudioConverter
    private var file: AVAudioFile?
    private var written: Int64 = 0
    static let sampleRate = 16_000

    init(url: URL, sampleRate: Int) throws {
        guard let source = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate), channels: 1, interleaved: false) else { throw Failure.invalidFormat }
        let settings: [String: Any] = [AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: Double(Self.sampleRate),
            AVNumberOfChannelsKey: 1, AVLinearPCMBitDepthKey: 16, AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false, AVLinearPCMIsNonInterleaved: false]
        let file = try AVAudioFile(forWriting: url, settings: settings, commonFormat: .pcmFormatInt16, interleaved: true)
        guard let converter = AVAudioConverter(from: source, to: file.processingFormat) else { throw Failure.invalidFormat }
        converter.primeMethod = .none
        self.source = source; self.converter = converter; self.file = file
        #if os(iOS)
        try FileManager.default.setAttributes([.protectionKey: FileProtectionType.completeUntilFirstUserAuthentication], ofItemAtPath: url.path)
        #endif
    }

    func append(_ samples: [Float]) throws {
        guard let file, !samples.isEmpty, let input = AVAudioPCMBuffer(pcmFormat: source, frameCapacity: AVAudioFrameCount(samples.count)),
              let channel = input.floatChannelData?[0] else { throw Failure.invalidFormat }
        input.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { pointer in channel.update(from: pointer.baseAddress!, count: samples.count) }
        let capacity = AVAudioFrameCount(ceil(Double(samples.count) * Double(Self.sampleRate) / source.sampleRate)) + 64
        guard let output = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: capacity) else { throw Failure.invalidFormat }
        var delivered = false; var error: NSError?
        let result = converter.convert(to: output, error: &error) { _, state in
            if delivered { state.pointee = .noDataNow; return nil }
            delivered = true; state.pointee = .haveData; return input
        }
        if let error { throw error }
        guard result != .error else { throw Failure.conversion }
        try write(output)
    }

    func finish() throws {
        guard let file else { return }
        guard let output = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: 4_096) else { throw Failure.invalidFormat }
        for _ in 0..<8 {
            var error: NSError?
            let result = converter.convert(to: output, error: &error) { _, state in state.pointee = .endOfStream; return nil }
            if let error { throw error }
            guard result != .error else { throw Failure.conversion }
            try write(output)
            if result == .endOfStream || output.frameLength == 0 { break }
        }
        self.file = nil
    }

    private func write(_ output: AVAudioPCMBuffer) throws {
        guard output.frameLength > 0 else { return }
        written += Int64(output.frameLength)
        guard written <= Int64(Self.sampleRate * 12 * 3_600 + 4_096) else { throw Failure.tooLong }
        try file?.write(from: output)
    }
}
#endif
