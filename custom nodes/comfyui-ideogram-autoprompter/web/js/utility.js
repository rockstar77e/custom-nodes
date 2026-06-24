const { app } = window.comfyAPI.app;

// ─── General-purpose helpers (adapted from ComfyUI-KJNodes) ───

export function chainCallback(object, property, callback) {
  if (object == undefined) {
    console.error("Tried to add callback to non-existant object");
    return;
  }
  if (property in object) {
    const callback_orig = object[property];
    object[property] = function () {
      const r = callback_orig.apply(this, arguments);
      callback.apply(this, arguments);
      return r;
    };
  } else {
    object[property] = callback;
  }
}

// ─── Middle-click pan passthrough for DOM widgets ───
export function addMiddleClickPan(element) {
  const onMouseDown = (e) => {
    if (e.button !== 1) return;
    e.preventDefault();
    const ds = app.canvas?.ds;
    if (!ds) return;
    const startX = e.clientX, startY = e.clientY;
    const startOffsetX = ds.offset[0], startOffsetY = ds.offset[1];
    const onMove = (me) => {
      ds.offset[0] = startOffsetX + (me.clientX - startX);
      ds.offset[1] = startOffsetY + (me.clientY - startY);
      app.canvas.setDirty(true, true);
    };
    const onUp = () => {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
    };
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  };
  element.addEventListener('mousedown', onMouseDown);
  return () => element.removeEventListener('mousedown', onMouseDown);
}

// ─── Wheel zoom passthrough for DOM widgets ───
export function addWheelPassthrough(element) {
  element.addEventListener("wheel", (e) => {
    const gc = document.getElementById("graph-canvas");
    if (gc) {
      gc.dispatchEvent(new WheelEvent(e.type, e));
      e.preventDefault();
    }
  }, { passive: false });
}

// Bounding-box hit test: corners, then edges, then interior.
export function rectHitTest(mx, my, x1, y1, x2, y2, radius) {
  const hit = (cx, cy) => Math.abs(mx - cx) < radius && Math.abs(my - cy) < radius;
  if (hit(x1, y1)) return "resize-tl";
  if (hit(x2, y1)) return "resize-tr";
  if (hit(x1, y2)) return "resize-bl";
  if (hit(x2, y2)) return "resize-br";
  if (mx >= x1 && mx <= x2 && Math.abs(my - y1) < radius) return "resize-t";
  if (mx >= x1 && mx <= x2 && Math.abs(my - y2) < radius) return "resize-b";
  if (my >= y1 && my <= y2 && Math.abs(mx - x1) < radius) return "resize-l";
  if (my >= y1 && my <= y2 && Math.abs(mx - x2) < radius) return "resize-r";
  if (mx >= x1 && mx <= x2 && my >= y1 && my <= y2) return "move";
  return null;
}

// CSS cursor for a bbox hit mode string.
export function cursorForBboxMode(mode) {
  if (mode === "move") return "move";
  if (mode === "resize-tl" || mode === "resize-br") return "nwse-resize";
  if (mode === "resize-tr" || mode === "resize-bl") return "nesw-resize";
  if (mode === "resize-t" || mode === "resize-b") return "ns-resize";
  if (mode === "resize-l" || mode === "resize-r") return "ew-resize";
  return null;
}
