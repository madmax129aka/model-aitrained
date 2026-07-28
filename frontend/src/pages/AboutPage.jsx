import React from "react";

export default function AboutPage() {
  return (
    <div className="flex flex-col gap-8 max-w-3xl">
      <div>
        <h1 className="text-2xl font-bold text-slate-100">About PixelTruth</h1>
        <p className="text-slate-500 mt-2 text-sm">
          PixelTruth estimates whether an image is likely AI-generated using a single
          signal: our own custom-trained CNN model. Here's a plain-language explanation,
          written for non-technical reviewers.
        </p>
      </div>

      <div className="glass-panel rounded-xl p-6">
        <h2 className="font-mono text-forensic-cyan text-sm uppercase tracking-wider mb-2">
          Our Custom CNN Model (100% of the score)
        </h2>
        <p className="text-sm text-slate-400 leading-relaxed">
          We trained a convolutional neural network from scratch on 100,000 labeled images
          (the CIFAKE dataset -- 50,000 real photos and 50,000 AI-generated images). The
          network looks at a resized 64x64 version of your image and learns visual
          patterns -- textures, edges, and color statistics -- that tend to differ between
          real camera photos and AI-generated ones. Every verdict PixelTruth gives you comes
          directly from this model's own prediction; there is no third-party AI-detection
          API involved anywhere.
        </p>
      </div>

      <div className="glass-panel rounded-xl p-6">
        <h2 className="font-mono text-forensic-cyan text-sm uppercase tracking-wider mb-2">
          Explainability Heatmap
        </h2>
        <p className="text-sm text-slate-400 leading-relaxed">
          Alongside the verdict, we show a saliency heatmap -- a visualization of which
          pixels most influenced the model's decision, computed via genuine gradient
          backpropagation through our own model's weights (not an approximation).
        </p>
      </div>

      <div className="glass-panel rounded-xl p-6 border border-forensic-amber/30">
        <h2 className="font-mono text-forensic-amber text-sm uppercase tracking-wider mb-2">
          A note on limitations
        </h2>
        <p className="text-sm text-slate-400 leading-relaxed">
          No AI detector is perfect. Our model reports 91% overall accuracy on a held-out
          test set of 20,000 images it never saw during training -- strong, but not
          infallible. The model was trained on the CIFAKE dataset, where "real" images come
          from a specific benchmark image collection; photos from other sources (e.g. a
          modern phone camera) can look statistically different to the model than anything
          it saw during training, which can affect accuracy on those images. Always treat
          PixelTruth's verdict as one piece of evidence, not a final legal or journalistic
          determination.
        </p>
      </div>
    </div>
  );
}
