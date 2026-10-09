# SWIFT-US fracture detector — browser demo

Static page: the YOLOv8n model (fold 0 of the cross-validation) runs in the visitor's browser with ONNX Runtime Web.
Nothing is uploaded; the page needs the network only to download `swiftus_fold0.onnx` (12 MB) and the runtime (11 MB) once.

## Files
- `index.html` — the page (all CSS/JS inline)
- `swiftus_fold0.onnx` — model exported from `cv5_yolov8n_img640_e100_s42/fold0/train/weights/best.pt` (`yolo export format=onnx imgsz=640 opset=17 simplify=True`)
- `ort.wasm.min.js`, `ort-wasm-simd-threaded.mjs`, `ort-wasm-simd-threaded.wasm` — ONNX Runtime Web 1.19.2 (MIT), wasm backend only
- `.nojekyll` — tells GitHub Pages to serve the files as they are

## Deploy on GitHub Pages
1. Create a repository (e.g. `swiftus-demo`), copy this folder's contents to its root, commit and push.
2. Repository → Settings → Pages → Source: "Deploy from a branch", branch `main`, folder `/ (root)`.
3. After a minute the page is at `https://<user>.github.io/swiftus-demo/`.
A custom domain (e.g. a UAlg subdomain) can be set in the same Pages settings.

## To update the model
Export the new weights to ONNX with the command above, replace `swiftus_fold0.onnx` (or change `MODEL_URL` in `index.html`), and push.

## Licences and permissions to settle before making the page public
- Ultralytics YOLOv8 is AGPL-3.0: a public deployment of weights trained with it has source-disclosure obligations (this demo's source is the page itself; publishing this folder as a public repository satisfies it) unless an Ultralytics Enterprise licence is obtained.
- The model was trained on images provided by Dr. O. Ackermann for exclusive use in the SWIFT-US project. Publishing a model derived from them is within the project, but confirm with him before making the page public.
- The page is labelled as a research prototype and makes no diagnostic claim; keep that wording.
