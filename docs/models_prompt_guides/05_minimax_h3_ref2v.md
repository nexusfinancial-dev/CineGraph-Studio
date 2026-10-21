# دليل هندسة البرومبت: MiniMax H3 (Ref2VA / Full-Reference Video)
**Model:** MiniMax Hailuo H3 (Video-01 / H3 Multimodal Engine)  
**Provider:** MiniMax  
**Primary Roles in Studio:** تحويل كادرات الكوميكس والصور المرجعية إلى لقطات فيديو سينمائية متحركة (Image/Ref-to-Video) مدعومة بالصوت المتزامن والمؤثرات وحركة الكاميرا (من 4 إلى 15 ثانية).

---

## 1. ما هو نمط Ref2VA في MiniMax H3؟ (Overview)

نمط **Ref2VA (Reference-to-Video-with-Audio)** هو النمط الأقوى والأشمل في معمارية MiniMax H3. على عكس التوليد البسيط من نص (T2V)، يسمح لك Ref2VA بـ:
1. تغذية الموديل بصورة الكادر المصمم أو كارت الشخصية (`<Picture 1>`).
2. تغذية الموديل بملف الصوت المُولد من Qwen Voice Design كمرجع صوتي (`<Audio 1>`).
3. إخراج فيديو سينمائي يتحرك فيه الكادر مع الحفاظ الكامل على ملامح الشخصية وحركة الشفاه وتزامن الصوت والبيئة.

---

## 2. الهيكلية الإلزامية للأقسام الستة (The 6-Section Framework)

يفرض موديل MiniMax H3 في وضع المراجع الكاملة (Full-Reference Mode) هيكلية قياسية صارمة من **6 أقسام متتالية بالترتيب**:

| القسم | الاسم التقني | الوظيفة |
| :---: | :--- | :--- |
| **1** | `subject_definitions` | تعريف المراجع والأصول بالأكواد (`<Subject N>`, `<Picture N>`, `<Audio N>`) |
| **2** | `summary` | ملخص نوع المهمة، زمن الفيديو، وطبيعة الحركة المطلوبة |
| **3** | `retention_analysis` | تحليل ما يتم الاحتفاظ به من المرجع وما يتغير أو يتحرك |
| **4** | `detailed_description` | السرد الزمني الدقيق للقطة ثانية بثانية مع الكاميرا والحوار |
| **5** | `overall_soundscape` | المؤثرات الصوتية والبيئية والطبيعية (Foley & Ambience) |
| **6** | `non_diegetic_music` | الموسيقى التصويرية الموجهة للمشاهد فقط |

---

## 3. قواعد وسوم المراجع (Reference Labels & Tags)

1. `<Subject N>`: يُستخدم لأي عنصر مرئي مستمر وملموس (شخصية، زي محدد، مبنى محدد).
2. `<Picture N>`: يُستخدم للصورة المرجعية التي تمثل الكادر الأساسي أو لقطة البداية (First Frame).
3. `<Audio N>`: يُستخدم للملف الصوتي المرجعي (صوت الحوار من Qwen Voice Design).
4. `<d>` و `(Sx)`: يُستخدم لتحديد نص الحوار المسموع وهوية المتحدث (مثال: `(S1) <d>I know someone's in there.</d>`).

---

## 4. مثال تطبيقي كامل جاهز للتشغيل في الاستوديو (A Warm Place - Scene 1)

هذا برومبت H3 Ref2VA رسمي ومطابق بنسبة 100% لمعايير MiniMax لإنتاج لقطة خروج ماري والش وتفاوضها مع كريس (مدة 6 ثوانٍ):

```text
subject_definitions:
<Picture 1> is the initial frame depicting the woman Mary Walsh standing in the hallway of an abandoned gas station in a heavy brown coat, crossing her arms.
<Subject 1> is Mary Walsh from <Picture 1>, a pale young woman in her early 20s with shoulder-length disheveled brown hair and sharp green eyes.
<Subject 2> is Chris Weston, a tall, broad-shouldered man in a distressed dark winter coat with a black eye on his left cheek, standing in profile foreground left.
<Audio 1> is the synchronized female dialogue track providing Mary's voice timbre and spoken lines.

summary:
Generate a 6-second cinematic drama dialogue shot continuing from <Picture 1>. Mary Walsh (<Subject 1>) stands up slightly straighter, maintains direct eye contact with Chris (<Subject 2>), and delivers her opening deal terms with synchronized lip motion matching <Audio 1>, while cold winter wind softly rattles the frosted windowpanes behind them.

retention_analysis:
- Mary Walsh's face, disheveled brown hair, oversized brown coat, and physical posture from <Picture 1> are fully preserved.
- The weathered gas station interior hallway and peeling doorway from <Picture 1> serve as the continuous environment.
- The camera slightly dollies forward while maintaining character continuity.
- <Audio 1> voice delivery directly drives Mary's facial articulation and lip synchronization.

detailed_description:
[00:00 - 00:03]
The shot opens matching <Picture 1>. Medium-shot composition. Mary Walsh (<Subject 1>) stands near the doorway, arms crossed over her chest. She takes a deliberate breath, visible vapor misting in the freezing interior air. As she begins speaking, her shoulders square slightly and her head tilts up with a coy, calculating expression. Chris Weston (<Subject 2>) is framed in the left foreground shoulder silhouette, listening motionlessly.
(S1) <d>You take care of me, get me food, keep me warm and safe...</d>
Camera slowly dollies inward toward Mary by 10%, keeping focus crisp on her green eyes.

[00:03 - 00:06]
Mary's gaze remains locked onto Chris with subtle confidence. A faint, almost imperceptible smirk forms at the corner of her lips as she finishes her terms. In the foreground, Chris's head gives a tiny amused nod. The wind outside causes a loose piece of sheet metal to clatter faintly against the roof.
(S1) <d>...and ultimately get me to civilization. In return, I will be your mistress.</d>
Camera settles on a tight medium close-up of Mary's composed, resolute expression as her breath slowly dissipates.

overall_soundscape:
Distant desolate winter wind whistling around exterior corners, low groaning wood rafters, faint metallic clatter of exterior signage. Crisp crunch of dust particles under Mary's slight foot adjustment. Subtle resonance of cold indoor space.

non_diegetic_music:
A low, sparse ambient cello drone on a single sustained note, accompanied by a cold, gentle glass-like piano chime, building subtle emotional intrigue and tension without overwhelming the dialogue.
```

---

## 5. صيغ الكاميرا وحركتها في Detailed Description (Camera Grammar)

في قسم `detailed_description`، يفضل دائماً استخدام مصطلحات الكاميرا الفيزيائية الدقيقة:
- **حركة التقريب الهادئ:** `Camera slowly dollies forward by 15% along the central axis`.
- **حركة التتبع الجانبي:** `Smooth horizontal tracking shot following the boots crunching into the snow`.
- **حركة التمحور حول الشخصية:** `Subtle slow 20-degree orbital pan around the subject from profile to three-quarter view`.
- **تأكيد البرودة والتنفس:** اذكر دائماً: `Visible breath condensation mist puffs in rhythm with speech`.

---

## 6. جدول مقارنة أنماط MiniMax H3 الأخرى

| النمط | المدخلات | متى تستخدمه في الاستوديو؟ |
| :--- | :--- | :--- |
| **Ref2VA** | صور مرجعية + صوت + نصوص | **النمط الأساسي للإنتاج الكامل المتزامن مع الصوت والشخصيات.** |
| **I2VA** | صورة واحدة (First Frame) + نص | تحريك كادر فردي سريع يبدأ من أول ثانية في الصورة. |
| **FL2VA** | صورتين (First Frame + Last Frame) | لقطات الانتقال المحسوبة التي تبدأ بكادر وتنتهي بكادر آخر محدد. |
| **T2VA** | نص فقط | توليد لقطات B-Roll سريعة للمشاهد الطبيعية (عاصفة ثلجية، شجرة عارية). |
