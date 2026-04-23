"""
rag/knowledge_base.py
Seeds the medical knowledge base with structured clinical documents.
These represent summarized guidelines from WHO, ATS, and radiology references.
"""

MEDICAL_DOCUMENTS = [
    {
        "id": "pneumonia_001",
        "title": "Community-Acquired Pneumonia: Radiological Features",
        "category": "Pneumonia",
        "content": """
Pneumonia on chest X-ray typically presents with the following radiological features:
- Consolidation: Homogeneous opacification replacing normal lung aeration
- Air bronchograms: Visible air-filled bronchi within consolidated lung
- Lobar or segmental distribution in bacterial pneumonia
- Bilateral patchy infiltrates in atypical (Mycoplasma, viral) pneumonia
- Perihilar infiltrates in viral pneumonia
- Pleural effusion may be present in up to 40% of cases

Clinical correlation: Fever, productive cough, elevated WBC, elevated CRP.
Pathogens: Streptococcus pneumoniae (most common), Haemophilus influenzae, Klebsiella.

Treatment guidelines (ATS/IDSA):
- Mild outpatient: Amoxicillin 1g TID x 5 days OR Doxycycline 100mg BID
- Moderate inpatient: Beta-lactam + macrolide combination
- ICU/severe: Anti-pseudomonal beta-lactam + macrolide or fluoroquinolone
- Follow-up X-ray at 6-8 weeks to confirm resolution
""",
    },
    {
        "id": "covid19_001",
        "title": "COVID-19 Pneumonia: Chest X-ray and CT Characteristics",
        "category": "COVID-19",
        "content": """
COVID-19 pneumonia (SARS-CoV-2) shows characteristic radiological patterns:

Early stage (0-4 days):
- Normal or subtle ground-glass opacities (GGO) in peripheral/lower zones
- Unilateral or bilateral involvement

Progressive stage (5-8 days):
- Bilateral, multilobar ground-glass opacities
- Crazy-paving pattern on CT (GGO + interlobular septal thickening)
- Peripheral and posterior predominance (75% of cases)

Peak stage (9-13 days):
- Consolidation with or without GGO
- Bilateral and diffuse distribution
- 'White lung' appearance in severe ARDS

Severity scoring: CO-RADS classification 1-6
CO-RADS 1: Normal
CO-RADS 5: Typical COVID-19

Key differentiators from other viral pneumonia:
- Peripheral distribution (vs central in influenza)
- Lower lobe predominance
- Vascular thickening adjacent to GGO

WHO treatment protocols:
- Mild: Supportive care, isolation
- Moderate: Supplemental oxygen, Dexamethasone 6mg/day x 10 days
- Severe: ICU, prone positioning, mechanical ventilation if SpO2 <90%
- Antivirals: Remdesivir, Nirmatrelvir-ritonavir (Paxlovid)
""",
    },
    {
        "id": "tb_001",
        "title": "Pulmonary Tuberculosis: Radiological Diagnosis Guide",
        "category": "Tuberculosis",
        "content": """
Tuberculosis (Mycobacterium tuberculosis) radiological patterns:

Primary TB (first infection):
- Lower or middle lobe consolidation (Ghon focus)
- Ipsilateral hilar/mediastinal lymphadenopathy
- Pleural effusion in children

Post-primary (reactivation) TB:
- Upper lobe predominance (apical and posterior segments)
- Fibronodular infiltrates
- Cavitation (pathognomonic in 40-45% of cases)
- Tree-in-bud pattern (endobronchial spread)
- Satellite nodules
- Calcified granulomas

Miliary TB:
- Diffuse 1-3mm nodules throughout both lungs
- 'Snowstorm' appearance

Smear-negative TB diagnosis criteria:
- Clinical symptoms >2 weeks (cough, night sweats, weight loss, hemoptysis)
- TST or IGRA positive
- Radiological features consistent with TB
- Response to anti-TB therapy

WHO First-line Treatment (RHEZ regimen):
- 2 months: Rifampicin + Isoniazid + Ethambutol + Pyrazinamide (RHEZ)
- 4 months: Rifampicin + Isoniazid (RH) continuation phase
- MDR-TB: Bedaquiline + Linezolid + Clofazimine x 6 months (BPaL regimen)

Infection control: Airborne precautions, N95 mask, negative pressure room
""",
    },
    {
        "id": "pleural_effusion_001",
        "title": "Pleural Effusion: Types, Imaging, and Management",
        "category": "Pleural Effusion",
        "content": """
Pleural effusion is accumulation of fluid in the pleural space.

Radiological findings:
- Blunting of costophrenic angle (>200mL on PA film)
- Meniscus sign (concave upper border)
- Opacification of affected hemithorax
- Mediastinal shift away (large effusion >1000mL)
- Subpulmonary effusion: apparent elevation of hemidiaphragm
- Lateral decubitus view: fluid >10mm confirms free-flowing effusion

Transudates (protein <30g/L, LDH <200 IU/L):
- Heart failure (bilateral, larger on right)
- Liver cirrhosis (right-sided hepatic hydrothorax)
- Nephrotic syndrome
- Hypoalbuminemia

Exudates (Light's criteria):
- Pleural:serum protein >0.5 OR LDH >0.6
- Causes: Parapneumonic, malignancy, TB, pulmonary embolism

Management:
- Treat underlying cause
- Therapeutic thoracentesis if dyspneic (max 1.5L per session)
- Chest tube drainage for empyema
- Chemical pleurodesis for recurrent malignant effusion
- VATS (surgery) for loculated effusion
""",
    },
    {
        "id": "normal_001",
        "title": "Normal Chest X-ray: Systematic Reading Approach",
        "category": "Normal",
        "content": """
A systematic approach to reading chest X-rays (ABCDE method):

A - Airway:
- Trachea midline, bifurcation at T4-T5
- Carina angle <70 degrees
- No visible foreign bodies or narrowing

B - Bones:
- Symmetric ribs, no fractures
- Vertebrae aligned
- Shoulder girdle intact
- No lytic or sclerotic lesions

C - Cardiac:
- Cardiothoracic ratio <0.5 on PA view
- Heart borders well-defined
- Aortic knuckle visible
- No mediastinal widening (>8cm on AP)

D - Diaphragm:
- Right hemidiaphragm 1.5-2.5cm higher than left
- Sharp costophrenic angles bilaterally
- No free air under diaphragm

E - Everything Else (Fields):
- Lung fields clear, no infiltrates
- Pulmonary vasculature to periphery
- No masses or nodules
- Hila at appropriate level
- Soft tissues symmetric

Normal variants:
- Cervical rib (1% of population)
- Pectus excavatum
- Scapular shadows

Technical adequacy assessment:
- Rotation: Clavicle heads equidistant from spinous process
- Inspiration: Anterior ends of 5-6 ribs above diaphragm
- Exposure: Thoracic vertebrae just visible through heart
""",
    },
    {
        "id": "radiology_safety_001",
        "title": "Radiation Safety and Imaging Protocols",
        "category": "General",
        "content": """
Radiation dose considerations in chest imaging:

Chest X-ray (PA view):
- Effective dose: 0.02-0.1 mSv
- Equivalent background radiation: ~10 days
- Safe for all ages including pregnancy (with shielding)

Chest CT:
- Effective dose: 4-7 mSv
- Equivalent background radiation: ~2 years
- Justify: Only when X-ray insufficient for diagnosis

ALARA principle (As Low As Reasonably Achievable):
- Use lowest dose that achieves diagnostic quality
- Prefer PA over AP (lower dose, better image quality)
- Digital detectors preferred (lower dose than film)
- Collimate to area of interest

Special populations:
- Pregnant women: Shield pelvis, use only when clinically essential
- Children: Use pediatric dose protocols (kilovoltage and mAs reduction)
- Repeat imaging: Document dose; avoid unnecessary repeats

AI-assisted diagnosis:
- FDA-cleared AI tools can flag critical findings (pneumothorax, PE)
- AI decision support does not replace radiologist interpretation
- Sensitivity 85-95%, Specificity 85-95% for major pathologies
- Always correlate with clinical history and labs
""",
    },
]


def get_all_documents() -> list[dict]:
    """Return all seeded medical documents."""
    return MEDICAL_DOCUMENTS
