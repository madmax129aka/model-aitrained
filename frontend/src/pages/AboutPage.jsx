import React from "react";

const SIGNALS = [
  {
    title: "1. Our Custom CNN Model (60% weight)",
    body:
      "We trained a convolutional neural network from scratch on the '140k Real and " +
      "Fake Faces' dataset -- 140,000 images total (100,000 for training, 20,000 for " +
      "validation, 20,000 held out for testing), perfectly balanced between real human " +
      "face photographs and AI-generated (StyleGAN) fake faces. The network looks at a " +
      "resized 128x128 version of your image and learns visual patterns -- textures, " +
      "edges, and facial structure -- that tend to differ between real photographed " +
      "faces and AI-generated ones. This is the most heavily weighted signal because it " +
      "is the only one that has actually studied thousands of labeled examples. " +
      "All inference runs on our own server -- no third-party AI-detection API is used.",
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

      <div className="glass-panel rounded-xl p-6 border border-forensic-cyan/30">
        <h2 className="font-mono text-forensic-cyan text-sm uppercase tracking-wider mb-2">
          Best results: photos with a clear human face
        </h2>
        <p className="text-sm text-slate-400 leading-relaxed">
          Our custom CNN was trained specifically on face photographs (real vs.
          AI-generated). Detection is most accurate on images that contain a clear human
          face, front-on or near-front-on. Non-face images (landscapes, objects,
          illustrations) can still be analyzed, but the CNN signal was not trained on that
          kind of content, so its accuracy there is less reliable than on faces -- the FFT
          and metadata signals still apply to any image type.
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
          No AI detector is perfect. Our model reports 94% overall accuracy on a held-out
          test set of 20,000 face images it never saw during training -- strong, but not
          infallible. Because the training data is faces-only, results on non-face images
          should be treated with more caution. Always treat PixelTruth's verdict as one
          piece of evidence, not a final legal or journalistic determination.
        </p>
      </div>
    </div>
  );
}
