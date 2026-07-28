import React from "react";

const SIGNALS = [
  {
    title: "1. Our Custom CNN Model (60% weight)",
    body:
      "We trained a convolutional neural network from scratch on 100,000 labeled images " +
      "(the CIFAKE dataset -- 50,000 real photos and 50,000 AI-generated images). The " +
      "network looks at a resized 64x64 version of your image and learns visual patterns " +
      "-- textures, edges, and color statistics -- that tend to differ between real camera " +
      "photos and AI-generated ones. This is the most heavily weighted signal because it is " +
      "the only one that has actually studied thousands of labeled examples.",
  },
  {
    title: "2. Frequency Domain (FFT) Analysis (20% weight)",
    body:
      "Every image can be described as a mix of frequencies, like a musical chord made of " +
      "different notes. AI image generators often build images through repeated " +
      "upsampling steps, which can leave behind faint, regular patterns invisible to the " +
      "eye but visible in the frequency spectrum. We compute this spectrum with plain math " +
      "(no AI involved) and check whether it looks unusually 'spiky' in a way that's " +
      "consistent with generated images, or smooth in a way that's typical of real photos.",
  },
  {
    title: "3. Metadata / EXIF Check (20% weight)",
    body:
      "Real cameras and phones usually embed metadata in a photo file -- the camera make " +
      "and model, exposure settings, sometimes even GPS location. AI-generated images " +
      "don't come from a camera, so they typically lack this information. We check for its " +
      "presence, but treat it as a weak clue: plenty of real photos lose this metadata too " +
      "(screenshots, social media re-uploads, edited images), so this signal alone never " +
      "decides the verdict.",
  },
];

export default function AboutPage() {
  return (
    <div className="flex flex-col gap-8 max-w-3xl">
      <div>
        <h1 className="text-2xl font-bold text-slate-100">About PixelTruth</h1>
        <p className="text-slate-500 mt-2 text-sm">
          PixelTruth combines three independent signals to estimate whether an image is
          likely AI-generated. Here's a plain-language explanation of each one, written for
          non-technical reviewers.
        </p>
      </div>

      <div className="flex flex-col gap-5">
        {SIGNALS.map((s) => (
          <div key={s.title} className="glass-panel rounded-xl p-6">
            <h2 className="font-mono text-forensic-cyan text-sm uppercase tracking-wider mb-2">
              {s.title}
            </h2>
            <p className="text-sm text-slate-400 leading-relaxed">{s.body}</p>
          </div>
        ))}
      </div>

      <div className="glass-panel rounded-xl p-6 border border-forensic-amber/30">
        <h2 className="font-mono text-forensic-amber text-sm uppercase tracking-wider mb-2">
          A note on limitations
        </h2>
        <p className="text-sm text-slate-400 leading-relaxed">
          No AI detector is perfect. Our model reports 91% overall accuracy on a held-out
          test set of 20,000 images it never saw during training -- strong, but not
          infallible. Always treat PixelTruth's verdict as one piece of evidence, not a
          final legal or journalistic determination.
        </p>
      </div>
    </div>
  );
}
