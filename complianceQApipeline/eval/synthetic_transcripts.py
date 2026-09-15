"""
synthetic_transcripts.py – Maps clip_id → synthetic transcript + OCR text.

Each entry faithfully represents the scenario described in ground_truth.csv so
that the pipeline's LLM auditor can reason about it correctly.  This decouples
the eval from needing real video files or Azure Video Indexer.

The batch runner imports CLIP_TRANSCRIPTS to inject pre-formed inputs directly
into the pipeline state (bypassing the index_video node).
"""

from typing import Dict, Any

# ---------------------------------------------------------------------------
# Each entry provides the fields that index_video would normally produce.
# ---------------------------------------------------------------------------

CLIP_TRANSCRIPTS: Dict[str, Dict[str, Any]] = {
    "clip_001": {
        "transcript": (
            "Hey guys, welcome back! Today I'm going to be talking about this amazing new skincare "
            "product I've been using. I absolutely love it and think you should try it too. "
            "Use my code SAVE20 for 20% off your first order. Link in bio!"
        ),
        "ocr_text": ["SAVE20 for 20% off", "Link in bio"],
        "visual_tags": ["skincare", "influencer", "product demo"],
        "duration_seconds": 62.0,
    },
    "clip_002": {
        "transcript": (
            "This video is sponsored by GlowBrand. Check out their new serum — it completely "
            "transformed my skin in just two weeks. I'm obsessed. Go to glowbrand.com now!"
        ),
        "ocr_text": ["[Ad]", "glowbrand.com"],
        "visual_tags": ["beauty", "serum", "influencer"],
        "duration_seconds": 45.0,
        "note": "The [Ad] label appears on screen for only 0.8 seconds in tiny text at the bottom.",
    },
    "clip_003": {
        "transcript": (
            "I wanted to tell you about this supplement I've been testing. It's incredible for "
            "energy. I take one every morning and feel amazing all day. Grab some at the link. "
            "Oh and by the way, this video was made in partnership with SupplementCo. #ad"
        ),
        "ocr_text": ["SupplementCo", "#ad"],
        "visual_tags": ["health", "supplements", "lifestyle"],
        "duration_seconds": 180.0,
        "note": "Disclosure only at the very end of a 3-minute video.",
    },
    "clip_004": {
        "transcript": (
            "Hi everyone! Disclaimer: This video is a paid partnership with FreshBrand — they "
            "sponsored this content. With that said, let me show you their new product line which "
            "I've been genuinely testing for 3 weeks. Here's my honest review..."
        ),
        "ocr_text": ["#ad", "#sponsored", "Paid Partnership with FreshBrand"],
        "visual_tags": ["product review", "lifestyle"],
        "duration_seconds": 540.0,
    },
    "clip_005": {
        "transcript": (
            "Before I start, I want to be fully transparent: this is a paid partnership with "
            "NutraBrand and I am being compensated for this content. Now let me show you "
            "what I honestly think about their protein powder..."
        ),
        "ocr_text": ["Paid Partnership", "NutraBrand"],
        "visual_tags": ["fitness", "protein", "partnership"],
        "duration_seconds": 300.0,
    },
    "clip_006": {
        "transcript": (
            "This supplement completely cured my diabetes in 30 days! I stopped taking my "
            "insulin and I feel incredible. The active ingredient reverses insulin resistance "
            "and regenerates pancreatic beta cells. Order now!"
        ),
        "ocr_text": ["Cures Diabetes", "Order Now", "Beta Cell Regeneration"],
        "visual_tags": ["health claims", "supplement", "medical"],
        "duration_seconds": 120.0,
    },
    "clip_007": {
        "transcript": (
            "Our software is 10x faster than our nearest competitor — period. We've benchmarked "
            "against every tool on the market and we always win. This is the absolute fastest "
            "solution available anywhere. No question."
        ),
        "ocr_text": ["10x FASTER", "Industry Leader", "Proven Performance"],
        "visual_tags": ["tech", "software", "benchmark"],
        "duration_seconds": 30.0,
    },
    "clip_008": {
        "transcript": (
            "Nine out of ten doctors recommend our VitaPlus supplement for daily immune support. "
            "Most healthcare professionals agree it's the best way to boost your health. "
            "Available now at VitaPlus.com!"
        ),
        "ocr_text": ["9/10 Doctors Recommend", "VitaPlus", "Immune Support"],
        "visual_tags": ["supplement", "health", "doctors"],
        "duration_seconds": 60.0,
    },
    "clip_009": {
        "transcript": (
            "This is simply the best protein powder on the market. Hands down the top-rated, "
            "most effective, number one choice for serious athletes. Nothing comes close. "
            "Try it today and see why it's the best."
        ),
        "ocr_text": ["#1 Best Seller", "Best Protein on Market"],
        "visual_tags": ["fitness", "protein", "supplement"],
        "duration_seconds": 45.0,
    },
    "clip_010": {
        "transcript": (
            "In our internal clinical study with 500 participants over 12 weeks, participants "
            "experienced an average of 15% improvement in energy levels. Individual results "
            "may vary. See our website for full study methodology and data."
        ),
        "ocr_text": ["*Based on 500-person internal study", "Results may vary", "See study details at brand.com"],
        "visual_tags": ["clinical study", "supplement", "energy"],
        "duration_seconds": 90.0,
    },
    "clip_011": {
        "transcript": (
            "In my personal experience, this moisturizer makes my skin feel incredibly soft. "
            "I love how it smells and I've been using it every morning. Just my opinion — "
            "everyone's skin is different and your results may vary!"
        ),
        "ocr_text": ["Personal Opinion", "Results Vary"],
        "visual_tags": ["skincare", "moisturizer", "lifestyle"],
        "duration_seconds": 75.0,
    },
    "clip_012": {
        "transcript": (
            "Let me show you how to use this NutraBrand protein shaker. You fill it up, shake it, "
            "and you're good to go. It's super convenient for the gym."
        ),
        "ocr_text": ["NutraBrand Shaker", "Gym Ready"],
        "visual_tags": ["fitness", "shaker", "CompetitorBrand logo visible in background"],
        "duration_seconds": 55.0,
        "note": "Competitor brand logo clearly visible on gym equipment in background throughout.",
    },
    "clip_013": {
        "transcript": (
            "What the f*** is this? This is absolute s***. Anyway, I'm wearing my new "
            "SportsBrand hoodie while I deal with this broken equipment. SportsBrand is "
            "sponsoring today's video."
        ),
        "ocr_text": ["SportsBrand", "#ad"],
        "visual_tags": ["sports", "clothing", "profanity"],
        "duration_seconds": 95.0,
    },
    "clip_014": {
        "transcript": (
            "Welcome to my chill evening routine! I've got my TechBrand laptop, my favorite "
            "snacks, and I'm just relaxing after a long day. This setup is perfect for unwinding."
        ),
        "ocr_text": ["TechBrand", "Evening Vibes"],
        "visual_tags": ["lifestyle", "relaxation", "alcohol bottles", "drug paraphernalia", "TechBrand laptop"],
        "duration_seconds": 120.0,
    },
    "clip_015": {
        "transcript": (
            "I just can't believe what's happening in politics right now. [Political rant]. "
            "Anyway, I'm drinking my HealthBrand protein shake while I vent about this. "
            "Use code SAVE for 15% off HealthBrand."
        ),
        "ocr_text": ["HealthBrand", "SAVE for 15% off"],
        "visual_tags": ["lifestyle", "politics", "protein shake"],
        "duration_seconds": 240.0,
    },
    "clip_016": {
        "transcript": (
            "Today I'm reviewing the ClearSkin face wash. I've been using it for 4 weeks. "
            "The packaging is clean, the scent is subtle, and it doesn't dry out my skin. "
            "Overall a solid product for everyday use."
        ),
        "ocr_text": ["ClearSkin Face Wash", "4-Week Review"],
        "visual_tags": ["skincare", "face wash", "clean studio"],
        "duration_seconds": 360.0,
    },
    "clip_017": {
        "transcript": (
            "Starting my morning with a LuxeWatch coffee and heading out for the day. "
            "Life feels good when you're wearing a LuxeWatch. Sponsored by LuxeWatch."
        ),
        "ocr_text": ["LuxeWatch", "Sponsored"],
        "visual_tags": ["lifestyle", "luxury", "morning routine"],
        "duration_seconds": 40.0,
    },
    "clip_018": {
        "transcript": (
            "Check out our new product in this exciting YouTube ad! Limited time offer — "
            "buy now and save 30%. Don't miss out on this incredible deal!"
        ),
        "ocr_text": ["SAVE 30%", "Limited Offer", "Buy Now"],
        "visual_tags": ["ad", "product", "4:3 aspect ratio"],
        "duration_seconds": 15.0,
        "note": "Video is encoded in 4:3 aspect ratio instead of required 16:9.",
    },
    "clip_019": {
        "transcript": (
            "Welcome to BrandX's new pre-roll ad! We're excited to share our latest innovation. "
            "This product will change your life. Visit brandx.com today for an exclusive offer!"
        ),
        "ocr_text": ["BrandX", "brandx.com", "Exclusive Offer"],
        "visual_tags": ["advertisement", "product launch"],
        "duration_seconds": 32.0,
        "note": "Duration is 32 seconds; pre-roll slots allow only 15s or 30s exactly.",
    },
    "clip_020": {
        "transcript": (
            "Use our app to track your fitness goals! Download now from the App Store "
            "or Google Play. Our premium plan gives you unlimited access to all features."
        ),
        "ocr_text": ["Download Now", "6pt font disclaimer: Subscription required. Auto-renews monthly."],
        "visual_tags": ["app ad", "fitness", "small text"],
        "duration_seconds": 15.0,
        "note": "Disclaimer text is 6pt — far below 30pt minimum for YouTube mobile ads.",
    },
    "clip_021": {
        "transcript": (
            "Introducing StreamPro — your ultimate streaming companion. Watch anything, "
            "anytime, anywhere. Subscribe today and get your first month free!"
        ),
        "ocr_text": ["StreamPro", "First Month Free", "SUBSCRIBE NOW (overlapping bottom safe zone)"],
        "visual_tags": ["streaming", "subscription", "ad"],
        "duration_seconds": 20.0,
        "note": "CTA button placed in bottom 15% violating YouTube safe zone spec.",
    },
    "clip_022": {
        "transcript": (
            "Discover the new PowerBlend blender! Crush ice, blend smoothies, and make "
            "soups in seconds. Order now at powerblend.com and use code BLEND20 for 20% off."
        ),
        "ocr_text": ["PowerBlend", "powerblend.com", "BLEND20 for 20% off", "1080p 16:9 Compliant Ad"],
        "visual_tags": ["kitchen appliance", "blender", "product ad"],
        "duration_seconds": 30.0,
    },
    "clip_023": {
        "transcript": (
            "GlowSerum changed my skin! #GlowSerum #skincaretips #ad"
        ),
        "ocr_text": ["#GlowSerum", "#ad", "9:16 vertical format"],
        "visual_tags": ["skincare", "shorts", "vertical video"],
        "duration_seconds": 58.0,
    },
    "clip_024": {
        "transcript": (
            "Obsessed with this new bag! So pretty. 🛍️ Get 15% off with my link!"
        ),
        "ocr_text": ["🛍️ 15% off link in bio", "❤️"],
        "visual_tags": ["fashion", "instagram story", "bag"],
        "duration_seconds": 15.0,
        "note": "Instagram-style story with no #ad/#sponsored disclosure whatsoever.",
    },
    # ── HOLDOUT SET ──────────────────────────────────────────────────────────
    "clip_025": {
        "transcript": (
            "FitPro Trainer burns exactly 500 calories per 30-minute session — guaranteed! "
            "That's a scientific fact proven by our proprietary FitScience technology. "
            "Order your FitPro Trainer today!"
        ),
        "ocr_text": ["Burns 500 calories GUARANTEED", "FitScience Technology", "Order Now"],
        "visual_tags": ["fitness", "exercise equipment", "calorie claim"],
        "duration_seconds": 30.0,
    },
    "clip_026": {
        "transcript": (
            "Good morning! This video is sponsored by MorningBrew Coffee. "
            "#ad #sponsored. I genuinely love this coffee and drink it every morning. "
            "Here's an honest review of their new blend..."
        ),
        "ocr_text": ["#ad", "#sponsored", "MorningBrew Coffee", "Sponsored Content"],
        "visual_tags": ["coffee", "morning routine", "sponsored"],
        "duration_seconds": 420.0,
    },
    "clip_027": {
        "transcript": (
            "Here's my morning wellness routine! I start with my VitaFlex supplements "
            "right after waking up. Combined with a healthy breakfast and these pills, "
            "I feel energized all day."
        ),
        "ocr_text": ["VitaFlex", "Morning Wellness"],
        "visual_tags": ["wellness", "supplement", "medication bottles visible", "painkiller packaging"],
        "duration_seconds": 90.0,
    },
    "clip_028": {
        "transcript": (
            "In my personal experience, this charcoal toothpaste makes my teeth feel "
            "noticeably whiter after 2 weeks of daily use. That's just my opinion — "
            "I'm not a dentist and your results may differ!"
        ),
        "ocr_text": ["Personal Review", "Not a dental endorsement"],
        "visual_tags": ["dental", "toothpaste", "lifestyle"],
        "duration_seconds": 180.0,
    },
    "clip_029": {
        "transcript": (
            "Get your energy drink fix with BurstDrink! Maximum flavor, maximum energy. "
            "Available at stores nationwide. BurstDrink — fuel your life!"
        ),
        "ocr_text": ["BurstDrink", "Fuel Your Life", "Available Nationwide"],
        "visual_tags": ["energy drink", "ad"],
        "duration_seconds": 15.0,
        "note": "Audio peak at -8 LUFS, exceeding YouTube -14 LUFS loudness ceiling.",
    },
    "clip_030": {
        "transcript": (
            "Easy 20-minute vegan meal prep with LeafBrand protein! Today I'm making "
            "chickpea curry and lentil soup. Let me show you how to batch cook healthy "
            "plant-based meals for the whole week. This video is sponsored by LeafBrand."
        ),
        "ocr_text": ["LeafBrand Protein", "Sponsored by LeafBrand", "#vegan", "#mealprep"],
        "visual_tags": ["vegan", "meal prep", "healthy cooking", "LeafBrand"],
        "duration_seconds": 780.0,
    },
    "clip_031": {
        "transcript": (
            "Loving this new palette! It's everything. The colors are stunning and it "
            "lasts all day. Grab yours using my link for 10% off!"
        ),
        "ocr_text": [
            "10% off at link in bio",
            "#makeup #beauty #glam #GRWM #tutorial #eyeshadow #motd #makeuplover "
            "#cosmetics #glowup #ad",
        ],
        "visual_tags": ["makeup", "beauty", "palette demo"],
        "duration_seconds": 95.0,
        "note": "#ad buried after 10 other hashtags at the very end of caption — not conspicuous.",
    },
    "clip_032": {
        "transcript": (
            "CloudSuite — work smarter."
        ),
        "ocr_text": ["CloudSuite", "cloudsuite.io", "Logo within safe zones"],
        "visual_tags": ["B2B SaaS", "bumper ad", "16:9", "6 seconds"],
        "duration_seconds": 6.0,
    },
}
