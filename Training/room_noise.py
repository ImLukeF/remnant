"""Reproducible waveform augmentation from verified, role-separated public noise clips."""
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path


class NoiseBank:
    def __init__(self,root):
        self.root=Path(root)
        self.manifest=json.loads((self.root/"bank.json").read_text())
        self.sha=hashlib.sha256((self.root/"bank.json").read_bytes()).hexdigest()
        if not self.manifest["approvedForAugmentation"]: raise ValueError("Noise licences have not been approved.")
        self.clips=self.manifest["clips"]
        roles={"train","validation","test"}; seen_sources=set(); seen_hashes=set()
        for clip in self.clips:
            if (clip["role"] not in roles or clip["kind"] not in {"speech","music","noise"}
                    or clip["licence"] not in {"CC0-1.0","CC-BY-3.0","CC-BY-4.0","Public-Domain"}
                    or clip["sourcePath"] in seen_sources or clip["sha256"] in seen_hashes):
                raise ValueError("Noise sources must be licensed and unique across all roles.")
            path=Path(clip["path"])
            if path.is_absolute() or ".." in path.parts: raise ValueError("Unsafe noise clip path.")
            seen_sources.add(clip["sourcePath"]);seen_hashes.add(clip["sha256"])
        self.roles={role:[i for i,c in enumerate(self.clips) if c["role"]==role] for role in roles}
        if any({self.clips[i]["kind"] for i in indexes}!={"speech","music","noise"} for indexes in self.roles.values()):
            raise ValueError("Each role needs independent speech, music and ambient clips.")

    @lru_cache(maxsize=64)
    def samples(self,index):
        import numpy as np
        clip=self.clips[index];path=self.root/clip["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest()!=clip["sha256"]: raise ValueError("Noise samples changed.")
        values=np.load(path,allow_pickle=False)
        if (values.ndim!=1 or not 30*16000<=len(values)<=60*16000 or values.dtype!=np.float32
                or not np.isfinite(values).all() or (np.abs(values)>1).any() or np.std(values.astype(np.float64))<1e-6):
            raise ValueError("Invalid bounded mono 16-kHz noise clip.")
        return values

    @lru_cache(maxsize=4)
    def resampled(self,index,rate):
        from scipy.signal import resample_poly
        values=self.samples(index).astype("float64")
        factor=math.gcd(16000,rate)
        return resample_poly(values,rate//factor,16000//factor) if rate!=16000 else values

    def mix(self,pcm,rate,subject,start,role):
        import numpy as np
        if role not in self.roles or not 8000<=rate<=192000 or len(pcm)!=30*rate:
            raise ValueError("Use one supported epoch and its preassigned noise role.")
        digest=hashlib.sha256(f"room-noise-v8:{subject}:{start:.6f}".encode()).digest()
        options=self.roles[role];index=options[int.from_bytes(digest[:8],"big")%len(options)]
        source=self.resampled(index,rate)
        offset=int.from_bytes(digest[8:16],"big")%(len(source)-len(pcm)+1)
        last=len(source)-len(pcm)
        for position in (offset,0,last//2,last):
            noise=source[position:position+len(pcm)].copy();noise-=noise.mean()
            noise_rms=float(np.sqrt(np.mean(noise*noise)))
            if noise_rms>=1e-9: break
        signal=np.asarray(pcm,dtype=np.float64);rms=float(np.sqrt(np.mean(signal*signal)))
        if not np.isfinite(signal).all() or noise_rms<1e-9: raise ValueError("Unusable augmentation waveform.")
        snr=(0,5,10,20)[digest[16]%4]
        mixed=signal+noise*(rms*10**(-snr/20)/noise_rms)
        peak=float(np.max(np.abs(mixed)))
        if peak>0.99: mixed*=0.99/peak
        return mixed,float(snr)
