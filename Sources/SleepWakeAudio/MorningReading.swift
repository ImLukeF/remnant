#if canImport(AVFAudio)
import AVFAudio
import Foundation
import SleepWakeKit

/// Reads a finished night's saved sound again with the morning model. Takes about a minute for a full night.
public enum MorningReading {
    public static func read(audio url: URL, night: SleepNight, model: MorningModel = .shipped) throws -> SleepNight.Morning? {
        guard night.observations.count >= MorningModel.minimumEpochs else { return nil }
        let file = try AVAudioFile(forReading: url)
        let format = file.processingFormat
        guard Int(format.sampleRate) == MorningSound.sampleRate, format.channelCount == 1,
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 64_000) else { return nil }
        var sound = MorningSound()
        while file.framePosition < file.length {
            try file.read(into: buffer)
            guard buffer.frameLength > 0, let samples = buffer.floatChannelData?[0] else { break }
            sound.append(UnsafeBufferPointer(start: samples, count: Int(buffer.frameLength)))
        }
        let rows = sound.rows(starts: night.observations.map { $0.start.timeIntervalSince(night.started) })
        let covered = rows.compactMap(\.self)
        guard covered.count >= MorningModel.minimumEpochs else { return nil }
        var stages = model.stages(covered).makeIterator()
        return .init(modelID: model.identifier, stages: rows.map { $0 == nil ? nil : stages.next() })
    }
}
#endif
