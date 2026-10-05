/**
 * SVM VisionLab - Clean User Interface Controller
 * 380x380 Canvas, Dedicated Detect Button, Essential Results Display.
 */

document.addEventListener("DOMContentLoaded", () => {
    // -------------------------------------------------------------
    // 1. Drawing Canvas (380x380)
    // -------------------------------------------------------------
    const canvas = document.getElementById("paintCanvas");
    const ctx = canvas.getContext("2d");
    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;
    const brushSlider = document.getElementById("brush-size");
    const clearBtn = document.getElementById("btn-clear-canvas");
    const detectBtn = document.getElementById("btn-detect-object");

    function resetCanvas() {
        ctx.fillStyle = "#000000";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = "#ffffff";
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.lineWidth = parseInt(brushSlider.value, 10);
    }
    resetCanvas();

    function getCanvasCoords(e) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;

        let clientX = e.clientX;
        let clientY = e.clientY;

        if (e.touches && e.touches.length > 0) {
            clientX = e.touches[0].clientX;
            clientY = e.touches[0].clientY;
        }

        return {
            x: (clientX - rect.left) * scaleX,
            y: (clientY - rect.top) * scaleY
        };
    }

    function startDraw(e) {
        e.preventDefault();
        isDrawing = true;
        const coords = getCanvasCoords(e);
        lastX = coords.x;
        lastY = coords.y;
    }

    function draw(e) {
        if (!isDrawing) return;
        e.preventDefault();
        const coords = getCanvasCoords(e);

        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = parseInt(brushSlider.value, 10);
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(coords.x, coords.y);
        ctx.stroke();

        lastX = coords.x;
        lastY = coords.y;
    }

    function stopDraw() {
        // Stop drawing without auto-triggering prediction!
        isDrawing = false;
    }

    canvas.addEventListener("mousedown", startDraw);
    canvas.addEventListener("mousemove", draw);
    canvas.addEventListener("mouseup", stopDraw);
    canvas.addEventListener("mouseleave", stopDraw);

    canvas.addEventListener("touchstart", startDraw, { passive: false });
    canvas.addEventListener("touchmove", draw, { passive: false });
    canvas.addEventListener("touchend", stopDraw);

    clearBtn.addEventListener("click", resetCanvas);

    brushSlider.addEventListener("input", () => {
        ctx.lineWidth = parseInt(brushSlider.value, 10);
    });

    // -------------------------------------------------------------
    // 2. Tabs Switcher
    // -------------------------------------------------------------
    const tabDraw = document.getElementById("tab-draw");
    const tabPresets = document.getElementById("tab-presets");
    const tabUpload = document.getElementById("tab-upload");

    const panelDraw = document.getElementById("panel-draw");
    const panelPresets = document.getElementById("panel-presets");
    const panelUpload = document.getElementById("panel-upload");

    let currentSourceMode = "canvas";

    function setInputMode(mode) {
        currentSourceMode = (mode === "upload") ? "upload" : "canvas";
        [tabDraw, tabPresets, tabUpload].forEach(t => {
            t.classList.remove("text-brand-400", "border-b-2", "border-brand-400");
            t.classList.add("text-slate-400");
        });
        [panelDraw, panelPresets, panelUpload].forEach(p => p.classList.add("hidden"));

        if (mode === "draw") {
            tabDraw.classList.add("text-brand-400", "border-b-2", "border-brand-400");
            tabDraw.classList.remove("text-slate-400");
            panelDraw.classList.remove("hidden");
        } else if (mode === "presets") {
            tabPresets.classList.add("text-brand-400", "border-b-2", "border-brand-400");
            tabPresets.classList.remove("text-slate-400");
            panelPresets.classList.remove("hidden");
        } else if (mode === "upload") {
            tabUpload.classList.add("text-brand-400", "border-b-2", "border-brand-400");
            tabUpload.classList.remove("text-slate-400");
            panelUpload.classList.remove("hidden");
        }
    }

    tabDraw.addEventListener("click", () => setInputMode("draw"));
    tabPresets.addEventListener("click", () => setInputMode("presets"));
    tabUpload.addEventListener("click", () => setInputMode("upload"));

    // -------------------------------------------------------------
    // 3. Preset Samples
    // -------------------------------------------------------------
    const presetsGrid = document.getElementById("presets-grid");
    const samplePresets = [
        { label: "👤 Human", type: "human" },
        { label: "🚗 Car", type: "car" },
        { label: "🌳 Tree", type: "tree" },
        { label: "🏠 House", type: "house" },
        { label: "🔤 Letter 'A'", type: "char", val: "A" },
        { label: "🔤 Letter 'B'", type: "char", val: "B" },
        { label: "🔢 Number '7'", type: "char", val: "7" },
        { label: "🔢 Number '3'", type: "char", val: "3" }
    ];

    function drawPresetToCanvas(tCtx, item, w, h) {
        tCtx.fillStyle = "#000000";
        tCtx.fillRect(0, 0, w, h);
        tCtx.fillStyle = "#ffffff";
        tCtx.strokeStyle = "#ffffff";
        const cx = w / 2;
        const cy = h / 2;
        const scale = w / 380;

        if (item.type === "human") {
            // Head
            tCtx.beginPath();
            tCtx.arc(cx, cy - 65*scale, 24*scale, 0, Math.PI * 2);
            tCtx.fill();
            // Torso & Limbs
            tCtx.lineWidth = 12*scale;
            tCtx.beginPath();
            tCtx.moveTo(cx, cy - 40*scale); tCtx.lineTo(cx, cy + 30*scale);
            tCtx.moveTo(cx - 45*scale, cy - 15*scale); tCtx.lineTo(cx + 45*scale, cy - 15*scale);
            tCtx.moveTo(cx, cy + 30*scale); tCtx.lineTo(cx - 35*scale, cy + 95*scale);
            tCtx.moveTo(cx, cy + 30*scale); tCtx.lineTo(cx + 35*scale, cy + 95*scale);
            tCtx.stroke();
        } else if (item.type === "char") {
            tCtx.font = `bold ${Math.round(w * 0.65)}px 'Inter', sans-serif`;
            tCtx.textAlign = "center";
            tCtx.textBaseline = "middle";
            tCtx.fillText(item.val, cx, cy + (w > 100 ? 10 : 2));
        } else if (item.type === "car") {
            tCtx.fillRect(cx - 70*scale, cy + 5*scale, 140*scale, 50*scale);
            tCtx.fillRect(cx - 35*scale, cy - 35*scale, 70*scale, 40*scale);
            tCtx.beginPath();
            tCtx.arc(cx - 42*scale, cy + 58*scale, 20*scale, 0, Math.PI * 2);
            tCtx.arc(cx + 42*scale, cy + 58*scale, 20*scale, 0, Math.PI * 2);
            tCtx.fill();
        } else if (item.type === "tree") {
            tCtx.fillRect(cx - 14*scale, cy + 15*scale, 28*scale, 85*scale);
            tCtx.beginPath();
            tCtx.arc(cx, cy - 25*scale, 65*scale, 0, Math.PI * 2);
            tCtx.fill();
        } else if (item.type === "house") {
            tCtx.fillRect(cx - 60*scale, cy - 15*scale, 120*scale, 90*scale);
            tCtx.beginPath();
            tCtx.moveTo(cx, cy - 85*scale);
            tCtx.lineTo(cx - 75*scale, cy - 15*scale);
            tCtx.lineTo(cx + 75*scale, cy - 15*scale);
            tCtx.closePath();
            tCtx.fill();
        }
    }

    function createPresetThumbnail(item) {
        const tempCanvas = document.createElement("canvas");
        tempCanvas.width = 64;
        tempCanvas.height = 64;
        const tCtx = tempCanvas.getContext("2d");
        drawPresetToCanvas(tCtx, item, 64, 64);
        const b64 = tempCanvas.toDataURL("image/png");

        const div = document.createElement("div");
        div.className = "cursor-pointer p-2 rounded-xl bg-slate-950 border border-slate-800 text-center hover:border-brand-400 transition";
        div.innerHTML = `
            <img src="${b64}" class="w-12 h-12 mx-auto rounded-lg mb-1 object-contain">
            <span class="text-[11px] text-slate-300 font-semibold block truncate">${item.label}</span>
        `;
        div.addEventListener("click", () => {
            drawPresetToCanvas(ctx, item, canvas.width, canvas.height);
            // Switch back to drawing tab so user can see it
            setInputMode("draw");
        });
        return div;
    }

    samplePresets.forEach(preset => {
        presetsGrid.appendChild(createPresetThumbnail(preset));
    });

    // -------------------------------------------------------------
    // 4. File Upload Handler
    // -------------------------------------------------------------
    const dropzone = document.getElementById("upload-dropzone");
    const fileInput = document.getElementById("image-file-input");
    const uploadPreview = document.getElementById("upload-preview");
    const uploadPreviewWrapper = document.getElementById("upload-preview-wrapper");

    dropzone.addEventListener("click", () => fileInput.click());

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFileUpload(e.target.files[0]);
        }
    });

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("border-brand-400");
    });
    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("border-brand-400");
    });
    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("border-brand-400");
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    function handleFileUpload(file) {
        const reader = new FileReader();
        reader.onload = (event) => {
            const rawB64 = event.target.result;
            uploadPreview.src = rawB64;
            uploadPreviewWrapper.classList.remove("hidden");

            const img = new Image();
            img.onload = () => {
                ctx.fillStyle = "#000000";
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            };
            img.src = rawB64;
        };
        reader.readAsDataURL(file);
    }

    // -------------------------------------------------------------
    // 5. Detect Object Button Trigger
    // -------------------------------------------------------------
    async function triggerDetection() {
        const imageBase64 = canvas.toDataURL("image/png");

        detectBtn.disabled = true;
        detectBtn.innerHTML = `<i data-lucide="loader" class="w-5 h-5 animate-spin"></i> Detecting & Classifying...`;
        if (window.lucide) lucide.createIcons();

        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    image: imageBase64,
                    source: currentSourceMode
                })
            });

            const data = await response.json();
            if (data.status === "success") {
                renderResults(data.result);
            }
        } catch (err) {
            console.error(err);
        } finally {
            detectBtn.disabled = false;
            detectBtn.innerHTML = `<i data-lucide="scan" class="w-5 h-5"></i> Detect Object / Classify`;
            if (window.lucide) lucide.createIcons();
        }
    }

    function renderResults(res) {
        const predTitle = document.getElementById("pred-title");
        const predIcon = document.getElementById("pred-badge-icon");
        const predConf = document.getElementById("pred-confidence");
        const predExpl = document.getElementById("pred-explanation");

        predTitle.textContent = res.predicted_class;
        predConf.textContent = `${res.confidence}%`;
        predExpl.textContent = res.explanation;

        // Choose appropriate icon
        const clsLower = res.predicted_class.toLowerCase();
        if (clsLower.includes("human") || clsLower.includes("person") || clsLower.includes("man") || clsLower.includes("woman")) {
            predIcon.textContent = "👤";
        } else if (clsLower.includes("car") || clsLower.includes("automobile")) {
            predIcon.textContent = "🚗";
        } else if (clsLower.includes("tree")) {
            predIcon.textContent = "🌳";
        } else if (clsLower.includes("house")) {
            predIcon.textContent = "🏠";
        } else if (clsLower.includes("letter") || clsLower.includes("digit")) {
            predIcon.textContent = res.predicted_class.slice(-1);
        } else {
            predIcon.textContent = "🎯";
        }

        // Render clean Top 3 candidate bars
        const candidatesContainer = document.getElementById("candidates-container");
        candidatesContainer.innerHTML = "";

        if (res.top_candidates && res.top_candidates.length > 0) {
            res.top_candidates.slice(0, 3).forEach((item, idx) => {
                const isTop = idx === 0;
                const div = document.createElement("div");
                div.className = "space-y-1";
                div.innerHTML = `
                    <div class="flex justify-between text-xs font-semibold ${isTop ? 'text-emerald-300 font-bold' : 'text-slate-400'}">
                        <span>${item.label}</span>
                        <span class="font-mono ${isTop ? 'text-emerald-400' : 'text-slate-400'}">${item.prob}%</span>
                    </div>
                    <div class="w-full bg-slate-950 rounded-full h-2 overflow-hidden border border-slate-800">
                        <div class="h-full rounded-full transition-all duration-300 ${isTop ? 'bg-emerald-400' : 'bg-slate-700'}" style="width: ${Math.max(4, item.prob)}%"></div>
                    </div>
                `;
                candidatesContainer.appendChild(div);
            });
        }
    }

    detectBtn.addEventListener("click", triggerDetection);

    // Initial Startup: Draw sample Human stick figure and detect
    drawPresetToCanvas(ctx, { type: "human" }, canvas.width, canvas.height);
    triggerDetection();
});
