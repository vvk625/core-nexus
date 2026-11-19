/*
 * jarvis-voice.js -- Zero-Roundtrip WebSocket Audio Queue Player
 */
(function () {
  "use strict";

  var muted = false;
  try { muted = localStorage.getItem("jarvisMuted") === "1"; } catch (e) {}

  var currentAudio = null;
  var audioQueue = [];
  var isPlaying = false;
  var sessionId = 0;

  function stop() {
    sessionId++;
    audioQueue = [];
    isPlaying = false;
    if (currentAudio) {
      try {
        currentAudio.pause();
        currentAudio.src = "";
      } catch (e) {}
      currentAudio = null;
    }
  }

  function base64ToBlobUrl(b64) {
    var bin = atob(b64);
    var len = bin.length;
    var bytes = new Uint8Array(len);
    for (var i = 0; i < len; i++) bytes[i] = bin.charCodeAt(i);
    var blob = new Blob([bytes], { type: "audio/mpeg" });
    return URL.createObjectURL(blob);
  }

  function drainQueue(mySession) {
    if (muted || mySession !== sessionId) return;
    if (isPlaying || !audioQueue.length) return;

    isPlaying = true;
    var b64 = audioQueue.shift();
    var url = base64ToBlobUrl(b64);
    var audio = new Audio(url);
    currentAudio = audio;

    function finish() {
      URL.revokeObjectURL(url);
      if (currentAudio === audio) currentAudio = null;
      isPlaying = false;
      drainQueue(mySession);
    }

    audio.onended = finish;
    audio.onerror = finish;
    audio.play().catch(finish);
  }

  function enqueueAudioChunk(b64) {
    if (muted || !b64) return;
    audioQueue.push(b64);
    drainQueue(sessionId);
  }

  async function speakDirect(text, agent) {
    if (muted || !text) return;
    stop();
    var mySession = sessionId;
    try {
      var res = await fetch("/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text, agent: agent || "JARVIS" })
      });
      if (!res.ok || mySession !== sessionId) return;
      var blob = await res.blob();
      if (!blob.size || mySession !== sessionId) return;
      var url = URL.createObjectURL(blob);
      var audio = new Audio(url);
      currentAudio = audio;
      audio.onended = function () { URL.revokeObjectURL(url); currentAudio = null; };
      await audio.play();
    } catch (e) {}
  }

  window.jarvisEnqueueAudio = enqueueAudioChunk;
  window.jarvisSpeak = speakDirect;
  window.jarvisStop = stop;

  function addButton() {
    var b = document.createElement("button");
    function label() {
      b.textContent = muted ? "🔇 VOICE: OFF" : "🔊 VOICE: ON";
      b.style.opacity = muted ? "0.65" : "1";
    }
    b.style.cssText =
      "position:fixed;right:28px;bottom:92px;z-index:99999;padding:7px 14px;" +
      "background:rgba(10,16,28,0.92);color:var(--accent, #10b981);border:1px solid var(--accent, #10b981);" +
      "border-radius:8px;font:700 11px/1 'Segoe UI',monospace;letter-spacing:1.5px;cursor:pointer;" +
      "box-shadow:0 0 12px var(--accent-glow, rgba(16,185,129,0.25));transition:all 0.25s;";
    b.onclick = function () {
      muted = !muted;
      try { localStorage.setItem("jarvisMuted", muted ? "1" : "0"); } catch (e) {}
      if (muted) stop();
      else speakDirect("Neural voice systems online, sir.", "JARVIS");
      label();
    };
    label();
    document.body.appendChild(b);
  }
  if (document.body) addButton();
  else document.addEventListener("DOMContentLoaded", addButton);
})();