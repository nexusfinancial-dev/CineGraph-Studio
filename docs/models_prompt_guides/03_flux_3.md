# دليل هندسة البرومبت: FLUX 3
**Model:** FLUX 3 (Multimodal Foundation Model)  
**Provider:** Black Forest Labs (BFL)  
**Primary Roles in Studio:** الرندرة الفائقة للكادرات النهائية (Final Panels)، التحكم المكاني الدقيق عبر المربعات المحددة (Bounding Boxes)، توليد النصوص واللافتات الصريحة، وتكوين صفحات الكوميكس متعددة الشخصيات.

---

## 1. الفلسفة المعمارية لـ FLUX 3 (Core Philosophy)

يمثل **FLUX 3** الجيل الأحدث من نماذج الانتشار والتحويل المتعدد الوسائط (Multimodal Transformer). يختلف جوهرياً عن الموديلات التقليدية بالآتي:
1. **الإلغاء التام للـ Negative Prompt:** الموديل لا يمتلك أي معالجة للبرومبت السلبي. إضافة أي كلمات سلبية في الواجهة أو الكود يتم تجاهلها تماماً، والاعتماد بالكامل على **الوصف الإيجابي الصريح**.
2. **وصف ما تراه، لا التزيين بالكلمات الرنانة:** مصطلحات مثل `hyperrealistic, 4K, photorealistic, masterpiece` غير مجدية. الجودة العالية تُستدعى عبر مواصفات الكاميرا والعدسة والإضاءة والخامة.
3. **نظام التحكم المكاني عبر الإحداثيات (Spatial Bounding Boxes):** ميزة ثورية تتيح لك تحديد مكان كل شخصية أو نص أو أداة بدقة البكسل على شبكة إحداثيات من `0` إلى `1000`.

---

## 2. ميزة الإحداثيات المكانية (Spatial Bounding Boxes Syntax)

تتيح لك هذه الميزة في FLUX 3 منع تداخل الشخصيات أو "انجراف" العناصر على الكادر، وتحديد مكان بالونات الحوار واللافتات:

### صيغة كتابة الإحداثيات:
تُكتب الإحداثيات بنظام: `[ymin, xmin, ymax, xmax]` على مقياس من `0` إلى `1000`:
- `ymin`: الحد العلوي (Top)
- `xmin`: الحد الأيسر (Left)
- `ymax`: الحد السفلي (Bottom)
- `xmax`: الحد الأيمن (Right)

### مثال تطبيقي في كادر لشخصيتين:
```text
A cinematic medium shot of two survivors inside an abandoned snowed-in gas station.
man[150, 100, 950, 480] is Chris Weston, a broad-shouldered man in a worn winter coat looking across with an amused smirk.
woman[220, 520, 950, 880] is Mary Walsh, standing up straight with her arms crossed, green eyes locked on him with a coy expression.
sign[80, 600, 200, 850] is an old metal sign that reads "ROADMAPS 50¢".
Cold overcast morning light pouring in through frosted windows on the left.
```

---

## 3. المعادلة القياسية لصياغة البرومبت (The FLUX 3 Formula)

$$\text{Subject} \rightarrow \text{Action / Pose} \rightarrow \text{Costume \& Physicality} \rightarrow \text{Spatial Layout} \rightarrow \text{Environment \& Lighting} \rightarrow \text{Camera Optics \& Medium}$$

1. **الموضوع الأساسي (Subject):** تحديد جنس الشخصية، عمرها التقريبي، وبنيتها.
2. **الفعل والتعبير (Action & Expression):** لغة الجسد، الاتجاه الحركي، تعبيرات الوجه بدقة.
3. **الملابس والعتاد (Wardrobe & Condition):** نوع الخامات (صوف، جلد مهترئ، جينز مغبر) ودرجة الاتساخ أو التجمد.
4. **الموقع والإضاءة (Environment & Lighting):** زاوية الضوء، درجة الحرارة اللونية (Kelvin)، الظلال.
5. **مواصفات الكاميرا والوسيط (Camera & Film):** بعد العدسة البؤري (35mm / 50mm / 85mm)، عمق الميدان (f/1.8 shallow depth of field)، ونوع الكادر.

---

## 4. كتابة النصوص الصريحة واليافطات (Text & Signage)
- ضع النص المراد كتابته حرفياً بين علامتي اقتباس `"..."`.
- حدد لون الخط، حجمه، وسماكته، وحالته المادية:
```text
A rusted metal gas pump partially buried in deep snow, featuring an embossed enamel plate that reads "UNLEADED" in faded white sans-serif lettering, with streaks of brown rust dripping down the numbers.
```

---

## 5. أمثلة عملية احترافية لإنتاج كادرات الكوميكس والسينما

### مثال 1: كادر سينمائي واقعي (Live-Action Cinema Still)
```text
Cinematic eye-level medium shot of Chris Weston walking through deep powdery snow. He has a prominent purple black eye on his left cheekbone, a healing split lower lip, and thick frost coating his dark stubble. He wears a heavy distressed canvas survival parka with a fur-lined hood down, heavy black snow boots crunching into snow, and carries a weathered backpack. Soft diffuse northern winter daylight, visible breath mist condensing in the freezing air, shot on Arri Alexa with a 50mm anamorphic lens at f/2.0, subtle film grain, rich muted color grade.
```

### مثال 2: كادر رواية مصورة (American Graphic Novel Comic Panel)
```text
Dynamic graphic novel comic panel depicting Mary Walsh stepping into the hallway. A pale young woman with shoulder-length disheveled brown hair and sharp anxious green eyes, wearing a heavy fleece-lined brown jacket and worn denim jeans. She crosses her arms over her chest, leaning slightly back against a wooden doorframe with a devious, calculating smirk. Inked comic illustration, bold expressive brush linework, high contrast noir chiaroscuro shadows, deep shadows and muted watercolor textures.
```

### مثال 3: كادر مقسم بتحديد الإحداثيات المكانية لمشهد الحوار (Multi-Subject Spatial Layout)
```text
A tense interior cinematic dialogue shot inside an empty gas station office.
man[180, 80, 950, 460] a rugged tall survivor standing in profile with an unholstered pistol pointed safely towards the floor, head tilted curiously.
woman[260, 540, 920, 880] a young woman with anxious green eyes peeking out from behind a half-open door, whispering nervously.
text_sign[100, 300, 180, 520] a peeling paper notice pinned to the wooden door that reads "EMPLOYEES ONLY".
Cold ambient blue light from the hallway contrasting with faint amber daylight from a skylight overhead.
```

---

## 6. قواعد الرندرة والـ Best Practices في FLUX 3

1. **لا تضع Negative Prompt نهائياً:** تأكد من إزالة عقدة الـ Negative في ComfyUI أو تركها فارغة تماماً لتوفير زمن الرندرة.
2. **استخدم أوصاف الإضاءة البديلة:**
   - بدلاً من `beautiful lighting`: استخدم `golden hour side-lighting with long soft shadows`.
   - بدلاً من `dramatic lighting`: استخدم `high-contrast Rembrandt lighting with deep falloff`.
3. **الدقة الموصى بها:**
   - الكادرات الأفقية (Cinematic 16:9): `1344 x 768` أو `1536 x 864`.
   - كادرات الكوميكس الرأسية (Comic Panel 9:16): `768 x 1344` أو `896 x 1536`.
   - الكادرات المربعة (Square 1:1): `1024 x 1024`.
