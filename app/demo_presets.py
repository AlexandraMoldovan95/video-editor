"""Pre-built example edit plans for the "Try a demo" section.

These are NOT live AI calls. They cost nothing and need no API key: each one is a complete,
already-validated EDL that gets loaded straight into the current project (same way a manually
edited plan does), so a visitor can instantly see three different editing "instructions" produce
three different videos from the exact same raw clips.

- EDITOR'S_NOTE_ENERGETIC_REEL / EDITOR'S_NOTE_CALM_VLOG below are hand-built by the developer to
  match what each preset's rules ask for (see app/presets.py) - they are honest, schema-valid
  example output, not a recorded AI run.
- TRAVEL_STORY is different: it's copied verbatim from a real `planner.py` run (clips AND the
  "reasoning" text), from early end-to-end testing of the travel_cinematic preset. It is genuine
  AI output, just replayed instead of regenerated, so this one specific run doesn't need to be
  paid for twice. (The original reasoning was written in Romanian, before the app's translation
  pass, and is reproduced here in an English translation that keeps the original meaning.)

Every clip below references the same small Pexels sample set described in `sources`. The "Try a
demo" button for a given preset is only enabled once a project has all of those files uploaded.
"""

SOURCES = [
    "16928481_1080_1920_50fps.mp4", "16581829_1440_2560_60fps.mp4", "15940078_2160_3840_30fps.mp4",
    "16615054_2160_3840_25fps.mp4", "16931601_1080_1920_120fps.mp4", "17198558_3840_2160_30fps.mp4",
    "12772832_1080_1920_30fps.mp4", "pexels-ayse-ela-dikye-49914919-16268987.jpg",
    "pexels-hayriyenur-351892318-39457179.jpg", "pexels-santesson89-37765467.jpg",
]

OUTPUT = {"width": 1080, "height": 1920, "fps": 30}

DEMO_PRESETS: dict[str, dict] = {
    "energetic_reel": {
        "label": "Energetic reel",
        "description": "\"Cut it fast, punchy color, one bold title.\" Hand-built example matching the reel_energetic preset's rules — not a live AI call.",
        "sources": SOURCES,
        "reasoning": "",
        "edl": {
            "output": OUTPUT,
            "clips": [
                {"source": "16931601_1080_1920_120fps.mp4", "in": 0.5, "out": 1.7, "fit": "crop_center",
                 "brightness": 0.0, "contrast": 1.2, "saturation": 1.15, "warmth": 0.0},
                {"source": "12772832_1080_1920_30fps.mp4", "in": 2.0, "out": 3.2, "fit": "crop_center",
                 "brightness": 0.0, "contrast": 1.2, "saturation": 1.15, "warmth": 0.0,
                 "transition_in": {"type": "slideleft", "duration": 0.3}},
                {"source": "16581829_1440_2560_60fps.mp4", "in": 1.0, "out": 2.3, "fit": "crop_center",
                 "brightness": 0.02, "contrast": 1.22, "saturation": 1.18, "warmth": 0.0,
                 "transition_in": {"type": "circleclose", "duration": 0.3}},
                {"source": "16615054_2160_3840_25fps.mp4", "in": 3.0, "out": 4.2, "fit": "crop_center",
                 "brightness": 0.0, "contrast": 1.2, "saturation": 1.15, "warmth": 0.0,
                 "transition_in": {"type": "wipeleft", "duration": 0.3}},
                {"source": "16928481_1080_1920_50fps.mp4", "in": 5.0, "out": 6.2, "fit": "crop_center",
                 "brightness": -0.02, "contrast": 1.2, "saturation": 1.15, "warmth": 0.0,
                 "transition_in": {"type": "smoothleft", "duration": 0.3}},
                {"source": "15940078_2160_3840_30fps.mp4", "in": 2.0, "out": 3.2, "fit": "crop_center",
                 "brightness": 0.0, "contrast": 1.2, "saturation": 1.15, "warmth": 0.0,
                 "transition_in": {"type": "circleopen", "duration": 0.3}},
                {"source": "17198558_3840_2160_30fps.mp4", "in": 1.0, "out": 2.2, "fit": "blur_pad",
                 "brightness": 0.0, "contrast": 1.2, "saturation": 1.15, "warmth": 0.0,
                 "transition_in": {"type": "slideup", "duration": 0.3}},
                {"source": "12772832_1080_1920_30fps.mp4", "in": 5.0, "out": 6.1, "fit": "crop_center",
                 "brightness": 0.0, "contrast": 1.2, "saturation": 1.15, "warmth": 0.0,
                 "transition_in": {"type": "fadewhite", "duration": 0.3}},
                {"source": "16931601_1080_1920_120fps.mp4", "in": 2.5, "out": 3.7, "fit": "crop_center",
                 "brightness": 0.0, "contrast": 1.2, "saturation": 1.15, "warmth": 0.0,
                 "transition_in": {"type": "slideleft", "duration": 0.3}},
                {"source": "16928481_1080_1920_50fps.mp4", "in": 8.0, "out": 9.1, "fit": "crop_center",
                 "brightness": -0.02, "contrast": 1.2, "saturation": 1.15, "warmth": 0.0,
                 "transition_in": {"type": "circleclose", "duration": 0.3}},
                {"source": "pexels-santesson89-37765467.jpg", "duration": 2.0, "motion": "zoom_in",
                 "fit": "crop_center", "brightness": 0.0, "contrast": 1.2, "saturation": 1.15, "warmth": 0.0,
                 "transition_in": {"type": "wipeleft", "duration": 0.3}},
            ],
            "texts": [
                {"content": "MAKE EVERY SECOND COUNT", "start": 0.0, "end": 2.2, "position": "bottom", "style": "title"},
            ],
        },
    },
    "calm_vlog": {
        "label": "Calm vlog",
        "description": "\"Long, gentle cuts, warm tone, a few lines of story.\" Hand-built example matching the vlog_calm preset's rules — not a live AI call.",
        "sources": SOURCES,
        "reasoning": "",
        "edl": {
            "output": OUTPUT,
            "clips": [
                {"source": "15940078_2160_3840_30fps.mp4", "in": 0.0, "out": 4.5, "fit": "crop_center",
                 "brightness": 0.02, "contrast": 1.05, "saturation": 1.1, "warmth": 0.08},
                {"source": "16928481_1080_1920_50fps.mp4", "in": 2.0, "out": 6.0, "fit": "crop_center",
                 "brightness": 0.02, "contrast": 1.05, "saturation": 1.1, "warmth": 0.08,
                 "transition_in": {"type": "fade", "duration": 0.7}},
                {"source": "17198558_3840_2160_30fps.mp4", "in": 0.5, "out": 4.5, "fit": "blur_pad",
                 "brightness": 0.03, "contrast": 1.05, "saturation": 1.08, "warmth": 0.07,
                 "transition_in": {"type": "dissolve", "duration": 0.7}},
                {"source": "12772832_1080_1920_30fps.mp4", "in": 1.0, "out": 5.0, "fit": "crop_center",
                 "brightness": 0.02, "contrast": 1.05, "saturation": 1.1, "warmth": 0.08,
                 "transition_in": {"type": "fade", "duration": 0.7}},
                {"source": "16581829_1440_2560_60fps.mp4", "in": 0.5, "out": 4.0, "fit": "crop_center",
                 "brightness": 0.02, "contrast": 1.05, "saturation": 1.1, "warmth": 0.08,
                 "transition_in": {"type": "dissolve", "duration": 0.7}},
                {"source": "16615054_2160_3840_25fps.mp4", "in": 0.5, "out": 4.0, "fit": "crop_center",
                 "brightness": 0.02, "contrast": 1.05, "saturation": 1.1, "warmth": 0.08,
                 "transition_in": {"type": "fade", "duration": 0.7}},
                {"source": "16931601_1080_1920_120fps.mp4", "in": 0.3, "out": 3.8, "fit": "crop_center",
                 "brightness": 0.02, "contrast": 1.05, "saturation": 1.1, "warmth": 0.08,
                 "transition_in": {"type": "fadeblack", "duration": 0.7}},
                {"source": "pexels-hayriyenur-351892318-39457179.jpg", "duration": 4.0, "motion": "zoom_out",
                 "fit": "blur_pad", "brightness": 0.03, "contrast": 1.05, "saturation": 1.08, "warmth": 0.07,
                 "transition_in": {"type": "fade", "duration": 0.7}},
                {"source": "15940078_2160_3840_30fps.mp4", "in": 8.0, "out": 12.0, "fit": "crop_center",
                 "brightness": 0.02, "contrast": 1.05, "saturation": 1.1, "warmth": 0.08,
                 "transition_in": {"type": "dissolve", "duration": 0.7}},
                {"source": "pexels-ayse-ela-dikye-49914919-16268987.jpg", "duration": 3.5, "motion": "zoom_in",
                 "fit": "crop_center", "brightness": 0.03, "contrast": 1.05, "saturation": 1.08, "warmth": 0.07,
                 "transition_in": {"type": "fade", "duration": 0.7}},
            ],
            "texts": [
                {"content": "Golden hour, slow mornings.", "start": 3.0, "end": 6.5, "position": "bottom", "style": "caption"},
                {"content": "Finding the quiet moments.", "start": 16.0, "end": 19.5, "position": "bottom", "style": "caption"},
                {"content": "The road goes on.", "start": 32.0, "end": 35.5, "position": "bottom", "style": "caption"},
            ],
        },
        "overrides": {"target_duration": 40},
    },
    "travel_story": {
        "label": "Cinematic travel story",
        "description": "A real plan Claude generated for this exact material, warm and cinematic (travel_cinematic preset) — replayed from a saved run instead of calling the API again.",
        "sources": SOURCES,
        "reasoning": (
            "We open with the whirling dervish (black & white, strong visual impact), then move through "
            "the warm, light-filled scenes (the forest with sun rays, the ferry at sunset, the horse at "
            "dusk) for rhythm, followed by the boat with Istanbul and the waterfall. A calm ending with "
            "the zoom-in photos and the horizontal clip on a blurred background. Cuts vary between 2 and "
            "3 seconds; total ≈15s. The color starts from a warm base, with small adjustments for the "
            "black-and-white clips (reduced saturation) and the darker ones (brightness)."
        ),
        "edl": {
            "output": OUTPUT,
            "clips": [
                {"source": "16928481_1080_1920_50fps.mp4", "in": 1.5, "out": 4.0, "motion": "zoom_in",
                 "fit": "crop_center", "brightness": 0.02, "contrast": 1.1, "saturation": 1.0, "warmth": 0.05},
                {"source": "16581829_1440_2560_60fps.mp4", "in": 1.0, "out": 3.8, "motion": "zoom_in",
                 "fit": "crop_center", "brightness": 0.02, "contrast": 1.05, "saturation": 1.1, "warmth": 0.08},
                {"source": "15940078_2160_3840_30fps.mp4", "in": 5.0, "out": 7.2, "motion": "zoom_in",
                 "fit": "crop_center", "brightness": 0.08, "contrast": 1.05, "saturation": 1.1, "warmth": 0.08},
                {"source": "pexels-hayriyenur-351892318-39457179.jpg", "duration": 2.5, "motion": "zoom_in",
                 "fit": "blur_pad", "brightness": 0.02, "contrast": 1.05, "saturation": 1.1, "warmth": 0.08},
                {"source": "16615054_2160_3840_25fps.mp4", "in": 3.0, "out": 5.5, "motion": "zoom_in",
                 "fit": "crop_center", "brightness": 0.03, "contrast": 1.05, "saturation": 1.1, "warmth": 0.08},
                {"source": "pexels-santesson89-37765467.jpg", "duration": 2.2, "motion": "zoom_in",
                 "fit": "crop_center", "brightness": 0.03, "contrast": 1.05, "saturation": 1.1, "warmth": 0.08},
            ],
            "texts": [],
        },
        "overrides": {"target_duration": 15, "color_look": "warm"},
    },
}
