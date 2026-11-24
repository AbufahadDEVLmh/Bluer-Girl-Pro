(() => {
  if (window.__bluerGirlLoaded) return;
  window.__bluerGirlLoaded = true;

  let settings = {
    enabled: true,
    streamer_safe_mode: true,
    show_hud: false,
    blur_mode: "gaussian",
    blur_intensity: 26,
    padding_pct: 10,
    sens_pct: 74,
    blur_face: true,
    blur_hair: true,
    blur_chest: true,
    blur_limbs: true,
    enable_anime_games: true,
    blur_fullbody: false,
    whitelisted_hosts: []
  };

  let emergencyFullVideoBlur = false;
  let serverOnline = true;
  let lastLatencyMs = 0;
  let nextVideoUid = 1;

  const manualBlurredElements = new Set();

  function isCurrentSiteWhitelisted() {
    const host = window.location.hostname;
    const list = Array.isArray(settings.whitelisted_hosts)
      ? settings.whitelisted_hosts
      : [];
    return list.includes(host);
  }

  function isActiveOnPage() {
    return Boolean(settings.enabled) && !isCurrentSiteWhitelisted();
  }

  chrome.storage.local.get(Object.keys(settings), (res) => {
    settings = { ...settings, ...res };
    updateHudDisplay();
    if (!isActiveOnPage()) clearAllOverlays();
  });

  chrome.storage.onChanged.addListener((changes) => {
    for (const [k, v] of Object.entries(changes)) {
      settings[k] = v.newValue;
    }
    updateHudDisplay();
    if (!isActiveOnPage()) {
      clearAllOverlays();
    } else {
      imageCache = new WeakMap();
      bgImageCache = new WeakMap();
      renderActiveBoxes();
    }
  });

  // Streamer Hotkeys:
  // Alt + B: Toggle emergency full-video blur on all videos
  // Alt + H: Toggle showing/hiding the bottom HUD status pill
  // Alt + Click: Manually blur/unblur any specific clicked element/video/image
  window.addEventListener("keydown", (e) => {
    if (e.altKey && (e.code === "KeyB" || e.key === "b" || e.key === "B")) {
      e.preventDefault();
      emergencyFullVideoBlur = !emergencyFullVideoBlur;
      updateHudDisplay();
      renderActiveBoxes();
    } else if (e.altKey && (e.code === "KeyH" || e.key === "h" || e.key === "H")) {
      e.preventDefault();
      settings.show_hud = !settings.show_hud;
      chrome.storage.local.set({ show_hud: settings.show_hud });
      updateHudDisplay();
    }
  });

  window.addEventListener(
    "click",
    (e) => {
      if (!e.altKey || !isActiveOnPage()) return;
      const target = e.target;
      if (!target || target.id === "bluergirl-streamer-hud") return;
      e.preventDefault();
      e.stopPropagation();
      const mediaEl =
        target.closest("video, img") ||
        target.querySelector("video, img") ||
        target;
      if (manualBlurredElements.has(mediaEl)) {
        manualBlurredElements.delete(mediaEl);
      } else {
        manualBlurredElements.add(mediaEl);
      }
      renderActiveBoxes();
    },
    true
  );

  let overlayRoot = null;
  let hudEl = null;
  let patchPool = [];

  function ensureOverlayRoot() {
    const parent = document.fullscreenElement || document.documentElement;
    if (!overlayRoot) {
      overlayRoot = document.createElement("div");
      overlayRoot.id = "bluergirl-overlay-root";
    }
    if (overlayRoot.parentNode !== parent) {
      parent.appendChild(overlayRoot);
    }
    if (!hudEl && window === window.top) {
      hudEl = document.createElement("div");
      hudEl.id = "bluergirl-streamer-hud";
      hudEl.title =
        "اضغط لقفل/فتح تمويه الطوارئ للفيديو (Alt + B) | اضغط ✕ لإخفاء الشريط (Alt + H)";
      hudEl.addEventListener("click", (e) => {
        e.stopPropagation();
        if (e.target && e.target.id === "bluergirl-hud-close") {
          settings.show_hud = false;
          chrome.storage.local.set({ show_hud: false });
          updateHudDisplay();
          return;
        }
        emergencyFullVideoBlur = !emergencyFullVideoBlur;
        updateHudDisplay();
        renderActiveBoxes();
      });
      parent.appendChild(hudEl);
      updateHudDisplay();
    } else if (hudEl && hudEl.parentNode !== parent && window === window.top) {
      parent.appendChild(hudEl);
    }
    return overlayRoot;
  }

  function updateHudDisplay() {
    if (!hudEl) return;
    if (!isActiveOnPage() || !settings.show_hud) {
      hudEl.style.display = "none";
      return;
    }
    hudEl.style.display = "flex";
    const closeBtnHtml = `<span id="bluergirl-hud-close" title="إخفاء الشريط (Alt+H)" style="margin-right:6px;padding:1px 6px;border-radius:50%;background:rgba(255,255,255,0.12);color:#D1D5DB;font-size:11px;cursor:pointer;">✕</span>`;
    if (emergencyFullVideoBlur) {
      hudEl.innerHTML = `<span style="color:#EF4444;">🔒 فزعة الطوارئ شغّالة (Alt+B عشان تفكها)</span>${closeBtnHtml}`;
      hudEl.style.borderColor = "rgba(239, 68, 68, 0.6)";
    } else if (!serverOnline) {
      hudEl.innerHTML = `<span style="color:#FBBF24;">⚠️ السيرفر طافي — شغّل البرنامج عشان يشتغل البلور</span>${closeBtnHtml}`;
      hudEl.style.borderColor = "rgba(251, 191, 36, 0.5)";
    } else {
      const msTxt = lastLatencyMs > 0 ? ` (${Math.round(lastLatencyMs)}ms)` : "";
      const safeTag = settings.streamer_safe_mode ? " • فزعة البث" : "";
      hudEl.innerHTML = `<span style="color:#34D399;">🛡️ Bluer Girl شغّال${safeTag}${msTxt}</span>${closeBtnHtml}`;
      hudEl.style.borderColor = "rgba(52, 211, 153, 0.35)";
    }
  }

  document.addEventListener("fullscreenchange", () => {
    ensureOverlayRoot();
    renderActiveBoxes();
  });

  function clearAllOverlays() {
    for (const p of patchPool) {
      p.style.display = "none";
    }
  }

  const sampleCanvas = document.createElement("canvas");
  const sampleCtx = sampleCanvas.getContext("2d", { willReadFrequently: true });

  let imageCache = new WeakMap();
  let bgImageCache = new WeakMap();
  const videoState = new WeakMap();

  const trackedImages = new Set();
  const trackedBgElements = new Set();
  const trackedVideos = new Set();

  function isElementVisibleInViewport(el) {
    if (!el || !el.isConnected) return false;
    const style = window.getComputedStyle(el);
    if (
      style.display === "none" ||
      style.visibility === "hidden" ||
      parseFloat(style.opacity) < 0.05
    ) {
      return false;
    }
    const r = el.getBoundingClientRect();
    if (r.width < 48 || r.height < 48) return false;
    if (
      r.bottom < 10 ||
      r.top > window.innerHeight - 10 ||
      r.right < 10 ||
      r.left > window.innerWidth - 10
    ) {
      return false;
    }
    return true;
  }

  function getMediaContentRect(el, intrinsicW, intrinsicH) {
    const r = el.getBoundingClientRect();
    if (!intrinsicW || !intrinsicH || intrinsicW <= 0 || intrinsicH <= 0) {
      return { left: r.left, top: r.top, width: r.width, height: r.height };
    }
    const style = window.getComputedStyle(el);
    const fit = style.objectFit || "fill";
    if (fit === "contain" || fit === "scale-down" || el.tagName === "VIDEO") {
      const elRatio = r.width / r.height;
      const mediaRatio = intrinsicW / intrinsicH;
      if (Math.abs(elRatio - mediaRatio) > 0.04) {
        if (mediaRatio > elRatio) {
          const contentH = r.width / mediaRatio;
          const offsetY = (r.height - contentH) * 0.5;
          return {
            left: r.left,
            top: r.top + offsetY,
            width: r.width,
            height: contentH
          };
        } else {
          const contentW = r.height * mediaRatio;
          const offsetX = (r.width - contentW) * 0.5;
          return {
            left: r.left + offsetX,
            top: r.top,
            width: contentW,
            height: r.height
          };
        }
      }
    }
    return { left: r.left, top: r.top, width: r.width, height: r.height };
  }

  function applyPatchVisualStyle(el) {
    const px = Math.max(10, Number(settings.blur_intensity) || 26);
    if (settings.blur_mode === "solid") {
      el.style.backdropFilter = "none";
      el.style.webkitBackdropFilter = "none";
      el.style.background = "rgba(22, 20, 30, 0.97)";
      el.style.border = "1px solid rgba(96, 165, 250, 0.25)";
    } else if (settings.blur_mode === "pixelate") {
      el.style.backdropFilter = `blur(${Math.max(12, Math.floor(px * 0.85))}px) contrast(145%)`;
      el.style.webkitBackdropFilter = `blur(${Math.max(12, Math.floor(px * 0.85))}px) contrast(145%)`;
      el.style.background =
        "repeating-linear-gradient(0deg, rgba(18,18,26,0.30) 0px, rgba(18,18,26,0.30) 4px, transparent 4px, transparent 8px), " +
        "repeating-linear-gradient(90deg, rgba(18,18,26,0.30) 0px, rgba(18,18,26,0.30) 4px, transparent 4px, transparent 8px)";
      el.style.border = "none";
    } else {
      el.style.backdropFilter = `blur(${px}px) saturate(110%)`;
      el.style.webkitBackdropFilter = `blur(${px}px) saturate(110%)`;
      el.style.background = "rgba(20, 20, 28, 0.18)";
      el.style.border = "none";
    }
  }

  function renderActiveBoxes() {
    if (!isActiveOnPage()) {
      clearAllOverlays();
      return;
    }

    ensureOverlayRoot();
    const screenRects = [];
    const now = performance.now();

    // 0. Manually blurred elements (Alt + Click)
    for (const el of manualBlurredElements) {
      if (!el.isConnected) {
        manualBlurredElements.delete(el);
        continue;
      }
      if (!isElementVisibleInViewport(el)) continue;
      const r = el.getBoundingClientRect();
      screenRects.push({ left: r.left, top: r.top, width: r.width, height: r.height });
    }

    // 1. Active video blur boxes (TikTok, YouTube, Reels, Twitch, Kick)
    for (const vid of trackedVideos) {
      if (!vid.isConnected) {
        trackedVideos.delete(vid);
        continue;
      }
      if (!isElementVisibleInViewport(vid)) continue;
      const cRect = getMediaContentRect(vid, vid.videoWidth, vid.videoHeight);

      if (emergencyFullVideoBlur) {
        screenRects.push({
          left: cRect.left,
          top: cRect.top,
          width: cRect.width,
          height: cRect.height
        });
        continue;
      }

      const st = videoState.get(vid);
      if (!st) continue;

      if (
        settings.streamer_safe_mode &&
        serverOnline &&
        !st.firstFrameDone &&
        !st.tainted &&
        now - st.createdAt < 280
      ) {
        screenRects.push({
          left: cRect.left,
          top: cRect.top,
          width: cRect.width,
          height: cRect.height
        });
        continue;
      }

      if (!st.boxes || st.boxes.length === 0) continue;

      st.boxes = st.boxes.filter((item) => now - item.ts <= 380);
      if (st.boxes.length === 0) continue;

      for (const item of st.boxes) {
        const [nx1, ny1, nx2, ny2] = item.box;
        const sx = cRect.left + nx1 * cRect.width;
        const sy = cRect.top + ny1 * cRect.height;
        const sw = (nx2 - nx1) * cRect.width;
        const sh = (ny2 - ny1) * cRect.height;
        if (sw >= 8 && sh >= 8) {
          screenRects.push({ left: sx, top: sy, width: sw, height: sh });
        }
      }
    }

    // 2. Active <img> blur boxes
    for (const img of trackedImages) {
      if (!img.isConnected) {
        trackedImages.delete(img);
        continue;
      }
      if (!isElementVisibleInViewport(img)) continue;
      const st = imageCache.get(img);
      if (!st || !st.boxes || st.boxes.length === 0) continue;

      const cRect = getMediaContentRect(img, img.naturalWidth, img.naturalHeight);
      for (const [nx1, ny1, nx2, ny2] of st.boxes) {
        const sx = cRect.left + nx1 * cRect.width;
        const sy = cRect.top + ny1 * cRect.height;
        const sw = (nx2 - nx1) * cRect.width;
        const sh = (ny2 - ny1) * cRect.height;
        if (sw >= 8 && sh >= 8) {
          screenRects.push({ left: sx, top: sy, width: sw, height: sh });
        }
      }
    }

    // 3. Active CSS background-image blur boxes
    for (const bgEl of trackedBgElements) {
      if (!bgEl.isConnected) {
        trackedBgElements.delete(bgEl);
        continue;
      }
      if (!isElementVisibleInViewport(bgEl)) continue;
      const st = bgImageCache.get(bgEl);
      if (!st || !st.boxes || st.boxes.length === 0) continue;

      const r = bgEl.getBoundingClientRect();
      for (const [nx1, ny1, nx2, ny2] of st.boxes) {
        const sx = r.left + nx1 * r.width;
        const sy = r.top + ny1 * r.height;
        const sw = (nx2 - nx1) * r.width;
        const sh = (ny2 - ny1) * r.height;
        if (sw >= 8 && sh >= 8) {
          screenRects.push({ left: sx, top: sy, width: sw, height: sh });
        }
      }
    }

    while (patchPool.length < screenRects.length && patchPool.length < 55) {
      const div = document.createElement("div");
      div.className = "bluergirl-patch";
      overlayRoot.appendChild(div);
      patchPool.push(div);
    }

    const count = Math.min(screenRects.length, patchPool.length);
    for (let i = 0; i < count; i++) {
      const r = screenRects[i];
      const p = patchPool[i];
      applyPatchVisualStyle(p);
      p.style.left = `${Math.round(r.left)}px`;
      p.style.top = `${Math.round(r.top)}px`;
      p.style.width = `${Math.round(r.width)}px`;
      p.style.height = `${Math.round(r.height)}px`;
      p.style.display = "block";
    }

    for (let i = count; i < patchPool.length; i++) {
      patchPool[i].style.display = "none";
    }
  }

  function computeNormIoU(b1, b2) {
    const ix1 = Math.max(b1[0], b2[0]);
    const iy1 = Math.max(b1[1], b2[1]);
    const ix2 = Math.min(b1[2], b2[2]);
    const iy2 = Math.min(b1[3], b2[3]);
    const iw = Math.max(0, ix2 - ix1);
    const ih = Math.max(0, iy2 - iy1);
    const inter = iw * ih;
    if (inter <= 0) return 0;
    const a1 = (b1[2] - b1[0]) * (b1[3] - b1[1]);
    const a2 = (b2[2] - b2[0]) * (b2[3] - b2[1]);
    return inter / Math.max(0.0001, a1 + a2 - inter);
  }

  function ensureVideoState(vid) {
    let st = videoState.get(vid);
    if (!st) {
      const uid = `vid_${nextVideoUid++}_${Math.random().toString(36).slice(2, 7)}`;
      st = {
        uid,
        boxes: [],
        busy: false,
        tainted: false,
        firstFrameDone: false,
        createdAt: performance.now(),
        lastSrc: vid.currentSrc || vid.src || "",
        listenersAttached: false
      };
      videoState.set(vid, st);
    }
    if (!st.listenersAttached) {
      st.listenersAttached = true;
      const resetHandler = () => {
        st.boxes = [];
        st.firstFrameDone = false;
        st.createdAt = performance.now();
        chrome.runtime.sendMessage({ type: "RESET_VIDEO", video_id: st.uid }, () => {
          void chrome.runtime.lastError;
        });
        renderActiveBoxes();
        analyzeVideoElement(vid);
      };
      vid.addEventListener("emptied", resetHandler);
      vid.addEventListener("loadeddata", resetHandler);
    }
    return st;
  }

  function updateVideoTrackedBoxes(vid, newNormBoxes) {
    const st = ensureVideoState(vid);
    st.firstFrameDone = true;
    const now = performance.now();

    if (!newNormBoxes || newNormBoxes.length === 0) {
      st.boxes = [];
      return;
    }

    const active = st.boxes.filter((item) => now - item.ts <= 350);
    const used = new Set();
    const updated = [];

    for (const nb of newNormBoxes) {
      let bestIdx = -1;
      let bestIou = 0;
      for (let i = 0; i < active.length; i++) {
        if (used.has(i)) continue;
        const iou = computeNormIoU(nb, active[i].box);
        if (iou > bestIou && iou > 0.15) {
          bestIou = iou;
          bestIdx = i;
        }
      }
      if (bestIdx >= 0) {
        used.add(bestIdx);
        const ob = active[bestIdx].box;
        const a = 0.72;
        updated.push({
          box: [
            ob[0] * (1 - a) + nb[0] * a,
            ob[1] * (1 - a) + nb[1] * a,
            ob[2] * (1 - a) + nb[2] * a,
            ob[3] * (1 - a) + nb[3] * a
          ],
          ts: now
        });
      } else {
        updated.push({ box: nb, ts: now });
      }
    }
    st.boxes = updated;
  }

  function analyzeVideoElement(vid) {
    if (!isActiveOnPage() || !isElementVisibleInViewport(vid)) return;
    if (vid.readyState < 2 || vid.videoWidth < 32 || vid.videoHeight < 32) return;

    const st = ensureVideoState(vid);
    const curSrc = vid.currentSrc || vid.src || "";
    if (curSrc && curSrc !== st.lastSrc) {
      st.boxes = [];
      st.firstFrameDone = false;
      st.createdAt = performance.now();
      st.lastSrc = curSrc;
      chrome.runtime.sendMessage({ type: "RESET_VIDEO", video_id: st.uid }, () => {
        void chrome.runtime.lastError;
      });
    }
    if (st.busy) return;
    st.busy = true;

    let dataUrl = null;
    if (!st.tainted) {
      const targetW = Math.min(480, vid.videoWidth);
      const targetH = Math.max(
        48,
        Math.round((vid.videoHeight / vid.videoWidth) * targetW)
      );
      sampleCanvas.width = targetW;
      sampleCanvas.height = targetH;
      try {
        sampleCtx.drawImage(vid, 0, 0, targetW, targetH);
        dataUrl = sampleCanvas.toDataURL("image/jpeg", 0.76);
      } catch (err) {
        st.tainted = true;
        dataUrl = null;
      }
    }

    if (dataUrl) {
      chrome.runtime.sendMessage(
        {
          type: "DETECT_IMAGE_DATA",
          imageBase64: dataUrl,
          video_id: st.uid,
          settings
        },
        (resp) => {
          st.busy = false;
          if (chrome.runtime.lastError || !resp) return;
          if (resp.offline) {
            serverOnline = false;
            st.boxes = [];
          } else if (resp.ok) {
            serverOnline = true;
            if (resp.ms) lastLatencyMs = resp.ms;
            updateVideoTrackedBoxes(vid, resp.boxes || []);
          }
          updateHudDisplay();
          renderActiveBoxes();
        }
      );
      return;
    }

    const cRect = getMediaContentRect(vid, vid.videoWidth, vid.videoHeight);
    const cropX = Math.max(0, cRect.left);
    const cropY = Math.max(0, cRect.top);
    const cropW = Math.min(window.innerWidth - cropX, cRect.width);
    const cropH = Math.min(window.innerHeight - cropY, cRect.height);

    if (cropW < 32 || cropH < 32) {
      st.busy = false;
      return;
    }

    chrome.runtime.sendMessage(
      {
        type: "CAPTURE_VIDEO_TAB",
        video_id: st.uid,
        crop: {
          x: cropX,
          y: cropY,
          w: cropW,
          h: cropH,
          vw: window.innerWidth,
          vh: window.innerHeight
        },
        settings
      },
      (resp) => {
        st.busy = false;
        if (chrome.runtime.lastError || !resp) return;
        if (resp.offline) {
          serverOnline = false;
          st.boxes = [];
        } else if (resp.ok) {
          serverOnline = true;
          if (resp.ms) lastLatencyMs = resp.ms;
          const mappedBoxes = (resp.boxes || []).map(([bx1, by1, bx2, by2]) => {
            const absX1 = cropX + bx1 * cropW;
            const absY1 = cropY + by1 * cropH;
            const absX2 = cropX + bx2 * cropW;
            const absY2 = cropY + by2 * cropH;
            return [
              Math.max(0, Math.min(1, (absX1 - cRect.left) / cRect.width)),
              Math.max(0, Math.min(1, (absY1 - cRect.top) / cRect.height)),
              Math.max(0, Math.min(1, (absX2 - cRect.left) / cRect.width)),
              Math.max(0, Math.min(1, (absY2 - cRect.top) / cRect.height))
            ];
          });
          updateVideoTrackedBoxes(vid, mappedBoxes);
        }
        updateHudDisplay();
        renderActiveBoxes();
      }
    );
  }

  function analyzeImageElement(img) {
    if (!isActiveOnPage() || !isElementVisibleInViewport(img)) return;
    if (!img.complete || img.naturalWidth < 48 || img.naturalHeight < 48) return;

    const currentSrc = img.currentSrc || img.src || "";
    if (!currentSrc) return;

    let st = imageCache.get(img);
    if (st && st.src === currentSrc) return;
    st = { src: currentSrc, boxes: [], busy: true };
    imageCache.set(img, st);

    const targetW = Math.min(480, img.naturalWidth);
    const targetH = Math.max(
      48,
      Math.round((img.naturalHeight / img.naturalWidth) * targetW)
    );
    sampleCanvas.width = targetW;
    sampleCanvas.height = targetH;

    let dataUrl = null;
    try {
      sampleCtx.drawImage(img, 0, 0, targetW, targetH);
      dataUrl = sampleCanvas.toDataURL("image/jpeg", 0.78);
    } catch (err) {
      dataUrl = null;
    }

    chrome.runtime.sendMessage(
      {
        type: "DETECT_IMAGE_DATA",
        imageBase64: dataUrl,
        imageUrl: dataUrl ? null : currentSrc,
        settings
      },
      (resp) => {
        st.busy = false;
        if (chrome.runtime.lastError || !resp) return;
        if (resp.offline) {
          serverOnline = false;
          imageCache.delete(img);
        } else if (resp.ok) {
          serverOnline = true;
          if (resp.ms) lastLatencyMs = resp.ms;
          st.boxes = resp.boxes || [];
        } else {
          imageCache.delete(img);
        }
        updateHudDisplay();
        renderActiveBoxes();
      }
    );
  }

  function extractCssBackgroundUrl(el) {
    const bg = window.getComputedStyle(el).backgroundImage;
    if (!bg || bg === "none" || !bg.includes("url(")) return null;
    const match = bg.match(/url\(["']?(https?:\/\/[^"')]+)["']?\)/i);
    return match ? match[1] : null;
  }

  function analyzeBgElement(bgEl) {
    if (!isActiveOnPage() || !isElementVisibleInViewport(bgEl)) return;
    const url = extractCssBackgroundUrl(bgEl);
    if (!url) return;

    let st = bgImageCache.get(bgEl);
    if (st && st.src === url) return;
    st = { src: url, boxes: [], busy: true };
    bgImageCache.set(bgEl, st);

    chrome.runtime.sendMessage(
      {
        type: "DETECT_IMAGE_DATA",
        imageBase64: null,
        imageUrl: url,
        settings
      },
      (resp) => {
        st.busy = false;
        if (chrome.runtime.lastError || !resp) return;
        if (resp.offline) {
          serverOnline = false;
          bgImageCache.delete(bgEl);
        } else if (resp.ok) {
          serverOnline = true;
          if (resp.ms) lastLatencyMs = resp.ms;
          st.boxes = resp.boxes || [];
        } else {
          bgImageCache.delete(bgEl);
        }
        updateHudDisplay();
        renderActiveBoxes();
      }
    );
  }

  function registerMediaElements() {
    document.querySelectorAll("video").forEach((v) => trackedVideos.add(v));
    document.querySelectorAll("img").forEach((img) => trackedImages.add(img));
    document
      .querySelectorAll('[style*="background-image"], [role="img"]')
      .forEach((el) => {
        if (el.tagName !== "IMG" && el.tagName !== "VIDEO") {
          trackedBgElements.add(el);
        }
      });
  }

  const observer = new MutationObserver(() => {
    registerMediaElements();
  });
  observer.observe(document.documentElement, { childList: true, subtree: true });
  registerMediaElements();

  setInterval(() => {
    if (!isActiveOnPage()) return;
    registerMediaElements();
    for (const vid of trackedVideos) {
      analyzeVideoElement(vid);
    }
    for (const img of trackedImages) {
      analyzeImageElement(img);
    }
    for (const bgEl of trackedBgElements) {
      analyzeBgElement(bgEl);
    }
    renderActiveBoxes();
  }, 65);

  window.addEventListener("scroll", renderActiveBoxes, {
    passive: true,
    capture: true
  });
  window.addEventListener("resize", renderActiveBoxes, { passive: true });
})();
