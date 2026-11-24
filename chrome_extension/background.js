const AI_SERVER_URL = "http://127.0.0.1:17890";

const DEFAULT_SETTINGS = {
  enabled: true,
  streamer_safe_mode: true, // Instant pre-blur on new TikTok/Reel/Short load while AI evaluates frame 1
  show_hud: false,          // Hidden by default so nothing appears on stream
  blur_mode: "gaussian",    // 'gaussian' | 'pixelate' | 'solid'
  blur_intensity: 26,       // px radius
  padding_pct: 10,
  sens_pct: 74,
  blur_face: true,
  blur_hair: true,
  blur_chest: true,
  blur_limbs: true,
  enable_anime_games: true,
  blur_fullbody: false
};

chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.local.get(Object.keys(DEFAULT_SETTINGS), (res) => {
    const toSet = { show_hud: false };
    for (const [k, v] of Object.entries(DEFAULT_SETTINGS)) {
      if (res[k] === undefined) {
        toSet[k] = v;
      }
    }
    chrome.storage.local.set(toSet);
  });
});

async function arrayBufferToBase64(buffer) {
  let binary = "";
  const bytes = new Uint8Array(buffer);
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunkSize));
  }
  return btoa(binary);
}

let lastCaptureTime = 0;
let cachedTabDataUrl = null;

async function getTabScreenshot(windowId) {
  const now = Date.now();
  // Reuse tab screenshot if requested within 45ms by multiple video elements on the same page
  if (cachedTabDataUrl && now - lastCaptureTime < 45) {
    return cachedTabDataUrl;
  }
  return new Promise((resolve) => {
    chrome.tabs.captureVisibleTab(
      windowId,
      { format: "jpeg", quality: 74 },
      (dataUrl) => {
        if (chrome.runtime.lastError || !dataUrl) {
          resolve(null);
        } else {
          lastCaptureTime = Date.now();
          cachedTabDataUrl = dataUrl;
          resolve(dataUrl);
        }
      }
    );
  });
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === "CHECK_SERVER") {
    fetch(`${AI_SERVER_URL}/health`, { method: "GET" })
      .then((r) => r.json())
      .then((data) =>
        sendResponse({
          online: true,
          hardware: data.hardware || "GPU",
          total_requests: data.total_requests || 0,
          cache_hits: data.cache_hits || 0,
          avg_ms: data.avg_ms || 0
        })
      )
      .catch(() => sendResponse({ online: false }));
    return true;
  }

  if (msg.type === "RESET_VIDEO") {
    fetch(`${AI_SERVER_URL}/detect`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reset_video: true, video_id: msg.video_id })
    }).catch(() => {});
    sendResponse({ ok: true });
    return false;
  }

  if (msg.type === "CAPTURE_VIDEO_TAB") {
    (async () => {
      try {
        const winId = sender && sender.tab ? sender.tab.windowId : null;
        const tabDataUrl = await getTabScreenshot(winId);
        if (!tabDataUrl) {
          sendResponse({ ok: false, boxes: [], offline: false });
          return;
        }

        const resp = await fetch(`${AI_SERVER_URL}/detect`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            image: tabDataUrl,
            crop: msg.crop,
            video_id: msg.video_id,
            settings: msg.settings || {}
          })
        });
        const data = await resp.json();
        sendResponse({
          ok: Boolean(data.ok),
          boxes: data.boxes || [],
          ms: data.ms || 0,
          offline: false
        });
      } catch (err) {
        sendResponse({ ok: false, boxes: [], offline: true });
      }
    })();
    return true;
  }

  if (msg.type === "DETECT_IMAGE_DATA") {
    (async () => {
      try {
        let b64 = msg.imageBase64;
        if (!b64 && msg.imageUrl) {
          const imgResp = await fetch(msg.imageUrl);
          const buf = await imgResp.arrayBuffer();
          b64 = await arrayBufferToBase64(buf);
        }
        if (!b64) {
          sendResponse({ ok: false, boxes: [], offline: false });
          return;
        }

        const resp = await fetch(`${AI_SERVER_URL}/detect`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            image: b64,
            video_id: msg.video_id || null,
            settings: msg.settings || {}
          })
        });
        const data = await resp.json();
        sendResponse({
          ok: Boolean(data.ok),
          boxes: data.boxes || [],
          ms: data.ms || 0,
          offline: false
        });
      } catch (err) {
        sendResponse({ ok: false, boxes: [], offline: true });
      }
    })();
    return true;
  }
});
