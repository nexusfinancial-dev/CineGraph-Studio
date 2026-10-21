# دليل هندسة البرومبت: Qwen-Image-2.1 & Qwen-Image-2.1-Turbo
**Model:** Qwen-Image-2.1 / Qwen-Image-2.1-Turbo (7B Diffusion Transformer)  
**Provider:** Alibaba Cloud / Qwen Team  
**Primary Roles in Studio:** توليد وتجميع الصور (Image Assembly)، دمج الأصول المتعددة (Multi-Reference)، عزل الشخصيات بخلفية شفافة أصلية (Native RGBA Transparency)، والتعديل الموضعي الدقيق للكادرات.

---

## 1. ما الذي يجعل Qwen-Image-2.1 استثنائياً؟ (Core Capabilities)

يمثل **Qwen-Image-2.1** نقلة نوعية في أدوات إنتاج الكوميكس والسينما بفضل 4 ميزات لا تجتمع في موديل واحد:
1. **تجميع المشاهد من مراجع متعددة (Multi-Reference Assembly):**
   - يدعم إدخال حتى **10 صور مرجعية في نفس الطلب**!
   - يمكنك تغذية الموديل بـ: (صورة وجه البطل كريس) + (صورة المعطف والعتاد) + (صورة محطة الوقود) ليقوم بتجميعها معاً في كادر واحد متجانس الإضاءة دون الحاجة لتدريب LoRA!
2. **الشفافية الأصلية الحقيقية (Native RGBA / Real Alpha Channel):**
   - الموديل يولد قنوات Alpha شفافة مباشرة دون الحاجة لأدوات تفريغ خلفية (RemoveBG). ممتاز لإنتاج "كروت الشخصيات" و"الأزياء" وقطع الكوميكس القابلة للتحريك.
3. **التعديل الموجه بالأوامر والتعليمات (Instruction-Based Editing):**
   - يقبل أوامر تعديل لغوية مباشرة: `"Change the expression to anxious"`, `"Add a black eye to the left cheek"`.
4. **دقة 2K وكتابة نصوص واضحة:**
   - يدعم دقة أصلية حتى 2048x2048 وكتابة نصوص ويافطات إنجليزية وصينية بدقة فائقة.

---

## 2. نسختي الموديل وإعدادات ComfyUI (Base vs. Turbo)

| الخاصية | النسخة الأساسية (Base) | نسخة التوربو (Turbo) |
| :--- | :--- | :--- |
| **عدد الخطوات (Steps)** | 25 – 35 خطوة | **8 خطوات فقط** |
| **مقياس التوجيه (CFG Scale)** | 4.0 – 6.0 | **1.0 (ثابت بدون CFG Guidance)** |
| **المشفر اللغوي (Text Encoder)** | Qwen3-VL-8B | Qwen3-VL-8B |
| **حجم الذاكرة (VRAM)** | متوفر بوزن BF16 ونسخ مكممة GGUF (Q4/Q8) | متوفر بوزن BF16 ونسخ GGUF خفيفة جداً |
| **الاستخدام في الاستوديو** | الرندرة التجميعية النهائية للكوميكس | التجميع السريع وتجربة تركيب المراجع |

---

## 3. صيغ البرومبت حسب الوظيفة (Prompt Formulas)

### الصيغة 1: التجميع من مراجع متعددة (Multi-Reference Assembly)
عند تزويد الموديل بصور مرجعية مرقمة (Image 1, Image 2, Image 3):

```text
Combine the elements from the provided reference images:
Take the facial features and rugged build of <Image 1> (Chris Weston).
Dress him in the heavy winter survival gear, distressed coat, and backpack from <Image 2>.
Place him inside the environment of <Image 3> (the abandoned snowy gas station).
Maintain uniform cold cinematic lighting and cohesive textures across the entire image.
```

### الصيغة 2: توليد أصل شفاف بدون خلفية (Native Transparent RGBA)
لاستخراج كروت الشخصيات والعتاد ووضعها فوق كادرات الكوميكس:

```text
Full-body concept art of [Character Name and Details], [Pose and Action], isolated on a pure transparent background, clean sharp silhouette edges, alpha channel transparency, no background elements, studio character turn-around lighting.
```

### الصيغة 3: التعديل الموضعي بالتعليمات (Instruction Editing)
لتعديل كادر موجود مسبقاً (Image-to-Image / Inpainting):

```text
Based on <Image 1>, keep the background and clothing exactly unchanged. Modify only the character's face: add a prominent dark purple black eye to his left cheekbone and a split lower lip with a trace of dried blood, changing his expression to a determined, exhausted stare.
```

---

## 4. أمثلة عملية جاهزة لمشروع *A Warm Place*

### مثال 1: كارت شخصية ماري والش بخلفية شفافة (Asset Bible Character Cutout)
```text
Full-body cinematic character sheet of Mary Walsh. A pale attractive young woman in her early 20s, average height, shoulder-length disheveled brown hair, anxious bright green eyes. Wearing an oversized heavy brown coat with fleece collar, faded denim jeans, and rugged winter snow boots. Standing in a slightly hesitant three-quarter pose, arms hugging herself against the cold. Isolated on a transparent background, clean alpha channel cutout, soft studio rim lighting highlighting the hair contours.
```

### مثال 2: تجميع مشهد الحوار والتفاوض بين كريس وماري (Scene Assembly)
```text
Cinematic medium-two shot composing the characters from the reference images:
The tall, broad-shouldered man from <Image 1> stands on the left in profile, his hand resting near his holstered pistol, looking toward the woman with a slight inquisitive smirk.
The woman from <Image 2> stands on the right, crossing her arms firmly over her chest, standing up straighter with a calculating, devious expression.
They stand in the dusty interior hallway of the gas station from <Image 3>, with snow drifting past broken frosted windows in the background. Low winter light casting elongated soft shadows on the linoleum floor.
```

### مثال 3: عتاد البطل المنفرد بخلفية شفافة (Inventory Card)
```text
Tactical post-apocalyptic survival backpack, weathered green-grey canvas with leather straps, bedroll strapped to the bottom, side pouch with dried rabbit meat strips wrapped in cloth, metal canteen attached to the shoulder harness. Isolated on a pure transparent background, pristine alpha cutout, sharp product lighting.
```

---

## 5. قواعد التحكم بالنصوص المكتوبة واليافطات (Text Rendering)
- ضع أي نص مكتوب تريده داخل علامتي تنصيص `"..."`.
- حدد مكان ونمط الخط ولون النص:
```text
A close-up of a frosted wooden sign hanging above the doorway, with carved yellow letters that read "MANAGEMENT ONLY", peeling paint and icicles hanging from the bottom edge.
```

---

## 6. نصائح التجميع (Compositing Best Practices)
1. **تجنب التناقض بين المراجع:** تأكد أن المرجع الأول (الوجه) والمرجع الثاني (الزي) لا يتعارضان مع زاوية الكاميرا المطلوبة.
2. **تحديد مصدر الضوء الموحد:** عند دمج عدة مراجع، اذكر دائماً جملة: `Harmonize lighting and shadows from the top-left window across all subjects`.
3. **في التوربو (Turbo):** استخدم دائماً `CFG = 1.0` و `Steps = 8`، ولا ترفع الـ CFG فوق 1.5 منعاً لتشوه البكسلات (Burned artifacts).
