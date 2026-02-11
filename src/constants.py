"""Constants and patterns for resume parsing and analysis."""

# ============================================================================
# Layout Detector - Multilingual Section Patterns
# ============================================================================

# Section patterns organized by category for better maintainability
# Each section has patterns for multiple languages
# Languages: English (en), French (fr), Spanish (es), German (de), Italian (it), Portuguese (pt)

MULTILINGUAL_SECTIONS = {
    'experience': [
        # English
        r'(?i)^\s*(experience|work experience|professional experience|employment|work history|career history|professional background)\s*$',
        # French
        r'(?i)^\s*(expérience|expérience professionnelle|expériences|parcours professionnel|expérience de travail|carrière professionnelle|historique professionnel)\s*$',
        # Spanish
        r'(?i)^\s*(experiencia|experiencia laboral|experiencia profesional|historial laboral|trayectoria profesional)\s*$',
        # German
        r'(?i)^\s*(berufserfahrung|arbeitserfahrung|beruflicher werdegang|tätigkeiten|berufliche erfahrung)\s*$',
        # Italian
        r'(?i)^\s*(esperienza|esperienza professionale|esperienze lavorative|carriera)\s*$',
        # Portuguese
        r'(?i)^\s*(experiência|experiência profissional|experiência de trabalho|histórico profissional)\s*$',
    ],
    'education': [
        # English
        r'(?i)^\s*(education|academic|academic background|educational background|degrees|qualifications|training)\s*$',
        # French
        r'(?i)^\s*(formation|formations|études|diplômes|parcours académique|formation académique|scolarité|cursus)\s*$',
        # Spanish
        r'(?i)^\s*(educación|formación|estudios|titulación|antecedentes académicos|preparación académica)\s*$',
        # German
        r'(?i)^\s*(ausbildung|bildung|studium|akademische ausbildung|schulbildung|hochschulbildung)\s*$',
        # Italian
        r'(?i)^\s*(istruzione|formazione|educazione|titoli di studio|percorso accademico)\s*$',
        # Portuguese
        r'(?i)^\s*(educação|formação|estudos|titulação|formação académica|formação acadêmica)\s*$',
    ],
    'skills': [
        # English
        r'(?i)^\s*(skills|technical skills|competencies|core skills|key skills|expertise|abilities|proficiencies)\s*$',
        # French
        r'(?i)^\s*(compétences|compétences clés|compétences techniques|savoir-faire|aptitudes|connaissances|informatique|technologies|outils|logiciels)\s*$',
        # Spanish
        r'(?i)^\s*(habilidades|competencias|habilidades técnicas|conocimientos|aptitudes|destrezas)\s*$',
        # German
        r'(?i)^\s*(fähigkeiten|kompetenzen|fachkenntnisse|technische fähigkeiten|kenntnisse|qualifikationen)\s*$',
        # Italian
        r'(?i)^\s*(competenze|abilità|competenze tecniche|conoscenze|capacità|abilità professionali)\s*$',
        # Portuguese
        r'(?i)^\s*(competências|habilidades|competências técnicas|conhecimentos|aptidões)\s*$',
    ],
    'summary': [
        # English
        r'(?i)^\s*(summary|objective|profile|professional summary|career objective|personal statement|about me)\s*$',
        # French
        r'(?i)^\s*(profil|résumé|objectif|objectif professionnel|présentation|présentation personnelle|à propos|description)\s*$',
        # Spanish
        r'(?i)^\s*(perfil|resumen|objetivo|objetivo profesional|presentación personal|acerca de mí)\s*$',
        # German
        r'(?i)^\s*(profil|zusammenfassung|ziel|berufsziel|persönliches profil|über mich)\s*$',
        # Italian
        r'(?i)^\s*(profilo|riassunto|obiettivo|obiettivo professionale|presentazione personale|chi sono)\s*$',
        # Portuguese
        r'(?i)^\s*(perfil|resumo|objetivo|objetivo profissional|apresentação pessoal|sobre mim)\s*$',
    ],
    'contact': [
        # English
        r'(?i)^\s*(contact|contact information|personal information|personal details)\s*$',
        # French
        r'(?i)^\s*(contact|coordonnées|informations personnelles|informations de contact|renseignements personnels)\s*$',
        # Spanish
        r'(?i)^\s*(contacto|información de contacto|información personal|datos personales)\s*$',
        # German
        r'(?i)^\s*(kontakt|kontaktinformationen|persönliche informationen|persönliche daten)\s*$',
        # Italian
        r'(?i)^\s*(contatto|informazioni di contatto|informazioni personali|dati personali)\s*$',
        # Portuguese
        r'(?i)^\s*(contacto|contato|informação de contacto|informações pessoais|dados pessoais)\s*$',
    ],
    'languages': [
        # English
        r'(?i)^\s*(languages|language skills|linguistic skills)\s*$',
        # French
        r'(?i)^\s*(langues|compétences linguistiques|maîtrise des langues)\s*$',
        # Spanish
        r'(?i)^\s*(idiomas|habilidades lingüísticas|conocimientos de idiomas)\s*$',
        # German
        r'(?i)^\s*(sprachen|sprachkenntnisse|sprachfähigkeiten)\s*$',
        # Italian
        r'(?i)^\s*(lingue|competenze linguistiche|conoscenze linguistiche)\s*$',
        # Portuguese
        r'(?i)^\s*(línguas|idiomas|competências linguísticas|conhecimentos linguísticos)\s*$',
    ],
    'certifications': [
        # English
        r'(?i)^\s*(certifications|certificates|professional certifications|licenses|accreditations)\s*$',
        # French
        r'(?i)^\s*(certifications|certificats|certifications professionnelles|habilitations|attestations)\s*$',
        # Spanish
        r'(?i)^\s*(certificaciones|certificados|certificaciones profesionales|licencias)\s*$',
        # German
        r'(?i)^\s*(zertifizierungen|zertifikate|berufliche zertifizierungen|lizenzen)\s*$',
        # Italian
        r'(?i)^\s*(certificazioni|certificati|certificazioni professionali|licenze)\s*$',
        # Portuguese
        r'(?i)^\s*(certificações|certificados|certificações profissionais|licenças)\s*$',
    ],
    'projects': [
        # English
        r'(?i)^\s*(projects|personal projects|academic projects|professional projects)\s*$',
        # French
        r'(?i)^\s*(projets|projets personnels|projets académiques|projets professionnels|réalisations)\s*$',
        # Spanish
        r'(?i)^\s*(proyectos|proyectos personales|proyectos académicos|proyectos profesionales)\s*$',
        # German
        r'(?i)^\s*(projekte|persönliche projekte|akademische projekte|berufliche projekte)\s*$',
        # Italian
        r'(?i)^\s*(progetti|progetti personali|progetti accademici|progetti professionali)\s*$',
        # Portuguese
        r'(?i)^\s*(projetos|projetos pessoais|projetos académicos|projetos acadêmicos|projetos profissionais)\s*$',
    ],
    'awards': [
        # English
        r'(?i)^\s*(awards|honors|distinctions|achievements|accomplishments|recognitions)\s*$',
        # French
        r'(?i)^\s*(récompenses|distinctions|prix|honneurs|réalisations|succès|accomplissements)\s*$',
        # Spanish
        r'(?i)^\s*(premios|honores|distinciones|logros|reconocimientos)\s*$',
        # German
        r'(?i)^\s*(auszeichnungen|ehren|auszeichnungen|erfolge|erkennungen)\s*$',
        # Italian
        r'(?i)^\s*(premi|onori|distinzioni|conquiste|riconoscimenti)\s*$',
        # Portuguese
        r'(?i)^\s*(prémios|prêmios|honras|distinções|conquistas|reconhecimentos)\s*$',
    ],
    'publications': [
        # English
        r'(?i)^\s*(publications|research|papers|articles|conferences)\s*$',
        # French
        r'(?i)^\s*(publications|recherches|articles|conférences|travaux de recherche)\s*$',
        # Spanish
        r'(?i)^\s*(publicaciones|investigación|artículos|conferencias)\s*$',
        # German
        r'(?i)^\s*(publikationen|forschung|artikel|konferenzen)\s*$',
        # Italian
        r'(?i)^\s*(pubblicazioni|ricerca|articoli|conferenze)\s*$',
        # Portuguese
        r'(?i)^\s*(publicações|investigação|pesquisa|artigos|conferências)\s*$',
    ],
    'interests': [
        # English
        r'(?i)^\s*(interests|hobbies|personal interests|activities)\s*$',
        # French
        r'(?i)^\s*(intérêts|centres d\'intérêt|loisirs|activités|passions)\s*$',
        # Spanish
        r'(?i)^\s*(intereses|aficiones|pasatiempos|actividades personales)\s*$',
        # German
        r'(?i)^\s*(interessen|hobbys|persönliche interessen|freizeitaktivitäten)\s*$',
        # Italian
        r'(?i)^\s*(interessi|hobby|interessi personali|attività)\s*$',
        # Portuguese
        r'(?i)^\s*(interesses|hobbies|interesses pessoais|atividades)\s*$',
    ],
    'references': [
        # English
        r'(?i)^\s*(references|professional references)\s*$',
        # French
        r'(?i)^\s*(références|références professionnelles|recommandations)\s*$',
        # Spanish
        r'(?i)^\s*(referencias|referencias profesionales)\s*$',
        # German
        r'(?i)^\s*(referenzen|berufliche referenzen)\s*$',
        # Italian
        r'(?i)^\s*(referenze|referenze professionali)\s*$',
        # Portuguese
        r'(?i)^\s*(referências|referências profissionais)\s*$',
    ],
    'volunteer_work': [
        # English
        r'(?i)^\s*(volunteer|volunteering|volunteer work|community service|volunteer experience)\s*$',
        # French
        r'(?i)^\s*(bénévolat|travail bénévole|bénévolat|service communautaire|implication sociale)\s*$',
        # Spanish
        r'(?i)^\s*(voluntariado|trabajo voluntario|servicio comunitario|voluntario)\s*$',
        # German
        r'(?i)^\s*(freiwilligenarbeit|ehrenamt|ehrenamtliche tätigkeit|gemeinschaftsdienst)\s*$',
        # Italian
        r'(?i)^\s*(volontariato|lavoro volontario|servizio comunitario|volontari)\s*$',
        # Portuguese
        r'(?i)^\s*(voluntariado|trabalho voluntário|serviço comunitário|voluntário)\s*$',
    ],
    'professional_affiliations': [
        # English
        r'(?i)^\s*(affiliations|professional affiliations|memberships|professional memberships|organizations|associations|societies)\s*$',
        # French
        r'(?i)^\s*(affiliations|adhésions|organisations|associations professionnelles|membres|appartenances)\s*$',
        # Spanish
        r'(?i)^\s*(afiliaciones|afiliaciones profesionales|membresías|organizaciones|asociaciones)\s*$',
        # German
        r'(?i)^\s*(mitgliedschaften|berufliche mitgliedschaften|organisationen|verbände|vereine)\s*$',
        # Italian
        r'(?i)^\s*(affiliazioni|affiliazioni professionali|appartenenze|organizzazioni|associazioni)\s*$',
        # Portuguese
        r'(?i)^\s*(afiliações|afiliações profissionais|membros|organizações|associações)\s*$',
    ],
    'speaking_engagements': [
        # English
        r'(?i)^\s*(speaking|speaking engagements|presentations|conference presentations|keynotes|lectures)\s*$',
        # French
        r'(?i)^\s*(conférences|présentations|interventions|présentations en conférence|discours)\s*$',
        # Spanish
        r'(?i)^\s*(conferencias|presentaciones|discursos|charlas|ponencias)\s*$',
        # German
        r'(?i)^\s*(vorträge|präsentationen|konferenzvorträge|reden|vorlesungen)\s*$',
        # Italian
        r'(?i)^\s*(conferenze|presentazioni|discorsi|relazioni|lezioni)\s*$',
        # Portuguese
        r'(?i)^\s*(conferências|apresentações|discursos|palestras)\s*$',
    ],
    'patents': [
        # English
        r'(?i)^\s*(patents|inventions|intellectual property|ip|patent applications)\s*$',
        # French
        r'(?i)^\s*(brevets|inventions|propriété intellectuelle|brevets déposés)\s*$',
        # Spanish
        r'(?i)^\s*(patentes|invenciones|propiedad intelectual|patentes solicitadas)\s*$',
        # German
        r'(?i)^\s*(patente|erfindungen|geistiges eigentum|patentanmeldungen)\s*$',
        # Italian
        r'(?i)^\s*(brevetti|invenzioni|proprietà intellettuale|brevetti depositati)\s*$',
        # Portuguese
        r'(?i)^\s*(patentes|invenções|propriedade intelectual|pedidos de patente)\s*$',
    ],
    'workshops': [
        # English
        r'(?i)^\s*(workshops|training|professional development|continuing education|seminars|courses)\s*$',
        # French
        r'(?i)^\s*(ateliers|formations|développement professionnel|formation continue|séminaires|cours)\s*$',
        # Spanish
        r'(?i)^\s*(talleres|capacitación|desarrollo profesional|educación continua|seminarios|cursos)\s*$',
        # German
        r'(?i)^\s*(workshops|schulungen|berufliche weiterbildung|fortbildung|seminare|kurse)\s*$',
        # Italian
        r'(?i)^\s*(workshop|formazione|sviluppo professionale|formazione continua|seminari|corsi)\s*$',
        # Portuguese
        r'(?i)^\s*(workshops|treinamento|desenvolvimento profissional|educação continuada|seminários|cursos)\s*$',
    ],
    'activities': [
        # English
        r'(?i)^\s*(activities|extracurricular activities|professional activities|sports|personal activities)\s*$',
        # French
        r'(?i)^\s*(activités|activités parascolaires|activités professionnelles|activités sportives)\s*$',
        # Spanish
        r'(?i)^\s*(actividades|actividades extracurriculares|actividades profesionales|actividades deportivas)\s*$',
        # German
        r'(?i)^\s*(aktivitäten|außerschulische aktivitäten|berufliche aktivitäten|sport)\s*$',
        # Italian
        r'(?i)^\s*(attività|attività extracurriculari|attività professionali|attività sportive)\s*$',
        # Portuguese
        r'(?i)^\s*(atividades|atividades extracurriculares|atividades profissionais|atividades desportivas)\s*$',
    ],
    'online_presence': [
        # English
        r'(?i)^\s*(online presence|digital profiles|social media|portfolio|links|websites|profiles)\s*$',
        # French
        r'(?i)^\s*(présence en ligne|profils numériques|réseaux sociaux|portfolio|liens|sites web)\s*$',
        # Spanish
        r'(?i)^\s*(presencia en línea|perfiles digitales|redes sociales|portafolio|enlaces|sitios web)\s*$',
        # German
        r'(?i)^\s*(online-präsenz|digitale profile|soziale medien|portfolio|links|websites)\s*$',
        # Italian
        r'(?i)^\s*(presenza online|profili digitali|social media|portfolio|link|siti web)\s*$',
        # Portuguese
        r'(?i)^\s*(presença online|perfis digitais|redes sociais|portfólio|links|sites)\s*$',
    ],
    'research': [
        # English
        r'(?i)^\s*(research|research experience|research interests|research projects|academic research)\s*$',
        # French
        r'(?i)^\s*(recherche|expérience de recherche|intérêts de recherche|projets de recherche)\s*$',
        # Spanish
        r'(?i)^\s*(investigación|experiencia de investigación|intereses de investigación|proyectos de investigación)\s*$',
        # German
        r'(?i)^\s*(forschung|forschungserfahrung|forschungsinteressen|forschungsprojekte)\s*$',
        # Italian
        r'(?i)^\s*(ricerca|esperienza di ricerca|interessi di ricerca|progetti di ricerca)\s*$',
        # Portuguese
        r'(?i)^\s*(pesquisa|investigação|experiência de pesquisa|interesses de pesquisa|projetos de pesquisa)\s*$',
    ],
    'exhibitions': [
        # English
        r'(?i)^\s*(exhibitions|art exhibitions|gallery exhibitions|shows|art shows|displays)\s*$',
        # French
        r'(?i)^\s*(expositions|expositions artistiques|expositions de galerie|salons|vernissages)\s*$',
        # Spanish
        r'(?i)^\s*(exposiciones|exposiciones de arte|exposiciones en galería|muestras)\s*$',
        # German
        r'(?i)^\s*(ausstellungen|kunstausstellungen|galerieausstellungen|schauen)\s*$',
        # Italian
        r'(?i)^\s*(mostre|mostre d\'arte|mostre in galleria|esposizioni)\s*$',
        # Portuguese
        r'(?i)^\s*(exposições|exposições de arte|exposições em galeria|mostras)\s*$',
    ],
    'productions': [
        # English
        r'(?i)^\s*(productions|credits|film credits|theater credits|performances|shows)\s*$',
        # French
        r'(?i)^\s*(productions|crédits|crédits film|crédits théâtre|spectacles)\s*$',
        # Spanish
        r'(?i)^\s*(producciones|créditos|créditos de película|créditos de teatro|espectáculos)\s*$',
        # German
        r'(?i)^\s*(produktionen|credits|film-credits|theater-credits|aufführungen)\s*$',
        # Italian
        r'(?i)^\s*(produzioni|credits|credits cinematografici|credits teatrali|spettacoli)\s*$',
        # Portuguese
        r'(?i)^\s*(produções|créditos|créditos de cinema|créditos de teatro|espetáculos)\s*$',
    ],
    'teaching': [
        # English
        r'(?i)^\s*(teaching|teaching experience|courses taught|lecturing|instruction|academic teaching)\s*$',
        # French
        r'(?i)^\s*(enseignement|expérience d\'enseignement|cours donnés|cours enseignés)\s*$',
        # Spanish
        r'(?i)^\s*(docencia|experiencia docente|cursos impartidos|enseñanza)\s*$',
        # German
        r'(?i)^\s*(lehre|lehrerfahrung|unterrichtete kurse|unterricht|akademische lehre)\s*$',
        # Italian
        r'(?i)^\s*(insegnamento|esperienza di insegnamento|corsi tenuti|didattica)\s*$',
        # Portuguese
        r'(?i)^\s*(ensino|experiência de ensino|cursos ministrados|docência)\s*$',
    ],
    'clinical_experience': [
        # English
        r'(?i)^\s*(clinical experience|clinical rotations|clinical training|medical experience|healthcare experience|patient care)\s*$',
        # French
        r'(?i)^\s*(expérience clinique|stages cliniques|formation clinique|soins aux patients)\s*$',
        # Spanish
        r'(?i)^\s*(experiencia clínica|rotaciones clínicas|formación clínica|cuidado del paciente)\s*$',
        # German
        r'(?i)^\s*(klinische erfahrung|klinische rotationen|klinische ausbildung|patientenversorgung)\s*$',
        # Italian
        r'(?i)^\s*(esperienza clinica|rotazioni cliniche|formazione clinica|assistenza ai pazienti)\s*$',
        # Portuguese
        r'(?i)^\s*(experiência clínica|rotações clínicas|formação clínica|cuidados ao paciente)\s*$',
    ],
    'technical_skills': [
        # English
        r'(?i)^\s*(technical skills|tech skills|technical competencies|technical expertise|software skills|programming skills)\s*$',
        # French
        r'(?i)^\s*(compétences techniques|aptitudes techniques|expertise technique|compétences informatiques)\s*$',
        # Spanish
        r'(?i)^\s*(habilidades técnicas|competencias técnicas|experticia técnica|habilidades de software)\s*$',
        # German
        r'(?i)^\s*(technische fähigkeiten|technische kompetenzen|technische expertise|software-kenntnisse)\s*$',
        # Italian
        r'(?i)^\s*(competenze tecniche|abilità tecniche|expertise tecnica|competenze informatiche)\s*$',
        # Portuguese
        r'(?i)^\s*(habilidades técnicas|competências técnicas|expertise técnica|habilidades de software)\s*$',
    ],
}

# Flatten all patterns into a single list for backwards compatibility
# Order: English (index 0), French (1), Spanish (2), German (3), Italian (4), Portuguese (5)
SECTION_PATTERNS = [
    pattern
    for patterns in MULTILINGUAL_SECTIONS.values()
    for pattern in patterns
]

# Language index mapping for accessing specific language patterns
# Used by LayoutDetector to get patterns for a specific language
LANGUAGE_INDICES = {
    'en': 0,
    'fr': 1,
    'es': 2,
    'de': 3,
    'it': 4,
    'pt': 5,
}

# ============================================================================
# Content Analysis Constants
# ============================================================================

# Strong action verbs for resume bullet points (English examples)
ACTION_VERBS = {
    'leadership': [
        'led', 'managed', 'directed', 'supervised', 'coordinated', 'orchestrated',
        'mentored', 'trained', 'guided', 'spearheaded', 'championed', 'pioneered',
    ],
    'technical': [
        'developed', 'engineered', 'architected', 'implemented', 'built', 'created',
        'designed', 'programmed', 'coded', 'debugged', 'optimized', 'refactored',
        'integrated', 'deployed', 'automated', 'configured', 'maintained',
    ],
    'business': [
        'analyzed', 'evaluated', 'researched', 'strategized', 'negotiated',
        'partnered', 'collaborated', 'delivered', 'launched', 'facilitated',
    ],
    'communication': [
        'presented', 'communicated', 'documented', 'reported', 'authored',
        'edited', 'published', 'translated', 'interpreted',
    ],
    'improvement': [
        'improved', 'enhanced', 'optimized', 'streamlined', 'revamped',
        'upgraded', 'transformed', 'modernized', 'innovated', 'refined',
    ],
}

# Weak verbs to avoid in resumes (English)
WEAK_VERBS = [
    'helped', 'assisted', 'supported', 'worked', 'participated', 'involved',
    'responsible', 'duties included', 'was', 'were', 'have been',
    'assisted with', 'worked on', 'helped with', 'tried', 'attempted',
    'used', 'utilized', 'did', 'made', 'got',
]

# ============================================================================
# ATS Scoring Constants
# ============================================================================

# Standard resume sections that ATS expects
EXPECTED_SECTIONS = [
    'experience',
    'education',
    'skills',
]

# Optional but beneficial sections
OPTIONAL_SECTIONS = [
    'summary',
    'contact',
    'languages',
    'certifications',
    'projects',
    'awards',
    'publications',
    'interests',
    'references',
    'volunteer_work',
    'professional_affiliations',
    'speaking_engagements',
    'patents',
    'workshops',
    'activities',
    'online_presence',
    'research',
    'exhibitions',
    'productions',
    'teaching',
    'clinical_experience',
    'technical_skills',
]

# Contact information fields
CONTACT_FIELDS = ['name', 'email', 'phone', 'linkedin', 'address', 'website', 'scholar', 'google scholar']
