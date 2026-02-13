"""Streamlit app for ATS Resume Analyzer - Phase 2 Enhanced."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
from parsers import PDFTextExtractor, LayoutDetector, SectionParser, aggregate_resume
from parsers.language_detector import LanguageDetector
from parsers.unified_extractor import UnifiedResumeExtractor
from parsers.skill_extractor import SkillExtractor, extract_skills_from_resume
from parsers.enhanced_ocr import extract_text_enhanced, PDFTextExtractorEnhanced

# Optional: LangExtract parser
try:
    from parsers.langextract_parser import LangExtractResumeParser, LANGEXTRACT_AVAILABLE
except ImportError:
    LANGEXTRACT_AVAILABLE = False
    LangExtractResumeParser = None
from scoring import ATSScorer
from scoring.enhanced_ats_scorer import EnhancedATSScorer
from analysis import (
    ContentAnalyzer,
    JobDescriptionParser,
    ResumeJobMatcher,
    RecommendationEngine,
    ATSSimulator,
    LLMClient,
    Priority,
)
from analysis.content_understanding import ContentUnderstandingEngine, analyze_resume_content
from analysis.advanced_job_matcher import (
    AdvancedJobMatcher,
    match_resume_to_job_advanced,
)
from utils import Config

# Optional: Supabase storage
try:
    from storage import SupabaseStore, SUPABASE_AVAILABLE
except Exception:
    SUPABASE_AVAILABLE = False
    SupabaseStore = None

# Page configuration
st.set_page_config(
    page_title="ATS Resume Analyzer - Phase 2",
    page_icon="📄",
    layout="wide",
)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'resume_text' not in st.session_state:
    st.session_state.resume_text = None
if 'parsed_resume' not in st.session_state:
    st.session_state.parsed_resume = None
if 'match_result' not in st.session_state:
    st.session_state.match_result = None
if 'ai_suggestions' not in st.session_state:
    st.session_state.ai_suggestions = None
if 'ai_suggestions_loading' not in st.session_state:
    st.session_state.ai_suggestions_loading = False

# Initialize LLM client
llm_client = None
llm_status = {'openrouter': {'available': False}, 'ollama': {'available': False}}
try:
    llm_client = LLMClient()
    llm_status = llm_client.get_status()
    if not (llm_status['openrouter']['available'] or llm_status['ollama']['available']):
        print("LLM initialized but no providers available")
except Exception as e:
    print(f"LLM initialization error: {e}")
    import traceback
    traceback.print_exc()

# Title
st.title("📄 ATS Resume Analyzer")

# Sidebar
st.sidebar.header("Options")
use_ocr = st.sidebar.checkbox("Use OCR (for image-based PDFs)", value=False)
show_raw_text = st.sidebar.checkbox("Show raw extracted text", value=False)
use_embeddings = st.sidebar.checkbox("Use semantic embeddings", value=False,
                                     help="Requires sentence-transformers")

# OCR Engine Selection (Phase 3)
st.sidebar.markdown("---")
st.sidebar.markdown("**OCR Engine (Phase 3)**")
ocr_engine_option = st.sidebar.selectbox(
    "Choose OCR engine",
    options=["Auto-detect (Recommended)", "PaddleOCR", "Tesseract", "EasyOCR", "PDF Native Only"],
    index=0,
    help="Auto-detect: Uses best available engine with automatic fallback chain"
)

# Extraction method selector
st.sidebar.markdown("---")
st.sidebar.markdown("**Extraction Method**")
extraction_options = ["Standard (Fast)", "Unified (Enhanced Structure)", "Aggregated (Best Quality)"]
if LANGEXTRACT_AVAILABLE:
    extraction_options.append("LangExtract (LLM-Powered)")
    st.sidebar.markdown("*✓ LangExtract available*")
else:
    st.sidebar.markdown("*LangExtract: pip install langextract*")

extraction_method = st.sidebar.selectbox(
    "Choose extraction method",
    options=extraction_options,
    index=1,  # Default to Unified
    help="Standard: Fast rule-based parsing. Unified: Enhanced structure detection. Aggregated: OCR+Unified+LangExtract merge. LangExtract: AI-powered detailed extraction (slower, requires API key)."
)

# LLM Status
# Industry Scoring Selection (Phase 3)
st.sidebar.markdown("---")
st.sidebar.markdown("**Scoring Mode (Phase 3)**")
industry_options = ["General", "Tech", "Creative", "Academic"]
selected_industry = st.sidebar.selectbox(
    "Choose scoring profile",
    options=industry_options,
    index=0,
    help="General: Standard ATS scoring. Tech: Emphasizes skills/projects. Creative: Allows design flexibility. Academic: Research-focused."
)
st.sidebar.markdown(f"*Using {selected_industry} scoring profile*")

st.sidebar.markdown("---")
st.sidebar.markdown("**LLM Available**")

if llm_status['openrouter']['available']:
    st.sidebar.markdown("✓ OpenRouter")

if llm_status['ollama']['available']:
    st.sidebar.markdown("✓ Ollama")
    
if not llm_status['openrouter']['available'] and not llm_status['ollama']['available']:
    st.sidebar.caption("Optional: Configure for AI features")


def generate_ai_suggestions(results, llm_client):
    """Generate AI suggestions for resume improvements.
    
    Args:
        results: The analysis results dictionary
        llm_client: The LLM client instance
        
    Returns:
        Dictionary containing AI suggestions
    """
    if not llm_client:
        return None
    
    suggestions = {
        'keywords': None,
        'improved_bullets': [],
        'enhancements': []
    }
    
    # Suggest keywords
    keyword_response = llm_client.suggest_keywords(results['text'])
    if keyword_response.success and keyword_response.text.strip():
        suggestions['keywords'] = keyword_response.text
    
    # Improve weak bullets
    all_bullets = results['content_quality'].bullet_points
    weak_bullets = [b for b in all_bullets if b.get('issue')]
    
    for bullet in weak_bullets[:3]:
        improved = llm_client.improve_bullet_point(bullet['text'])
        if improved.success and improved.text.strip():
            suggestions['improved_bullets'].append({
                'before': bullet['text'],
                'after': improved.text
            })
    
    # Enhance good bullets for high-scoring resumes
    good_bullets = [b for b in all_bullets if not b.get('issue')]
    ats_score_val = results.get('ats_score', {}).get('total_score', 0)
    
    if good_bullets and ats_score_val >= 75:
        import random
        random.seed(42)
        bullets_to_enhance = random.sample(good_bullets, min(2, len(good_bullets)))
        
        for bullet in bullets_to_enhance:
            enhanced = llm_client.enhance_bullet_critique(bullet['text'])
            if enhanced.success and enhanced.text.strip():
                suggestions['enhancements'].append({
                    'current': bullet['text'],
                    'enhancement': enhanced.text
                })
    
    return suggestions


# File upload
st.header("Upload Resume")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Reset AI suggestions when a new file is uploaded
    st.session_state.ai_suggestions = None
    st.session_state.ai_suggestions_loading = False
    
    # Save uploaded file temporarily
    temp_path = Config.PROCESSED_DIR / uploaded_file.name
    Config.PROCESSED_DIR.mkdir(exist_ok=True)
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    # Process the resume
    extractor = None
    with st.spinner("Processing resume..."):
        try:
            # Extract text - use enhanced OCR if selected
            if ocr_engine_option == "Auto-detect (Recommended)":
                # Use enhanced OCR with multi-engine fallback
                result = extract_text_enhanced(temp_path, use_ocr=use_ocr, min_confidence=0.7)
                
                # Show confidence info
                confidence = result.get("overall_confidence", 0)
                needs_review = result.get("needs_review", False)
                
                if needs_review:
                    st.warning(f"OCR confidence: {confidence:.1%} - Some text may need review")
                else:
                    st.success(f"OCR confidence: {confidence:.1%}")
                
                # Show uncertainty suggestions if any
                uncertainty = result.get("uncertainty_report", {})
                if uncertainty.get("suggestions"):
                    with st.expander("OCR Processing Suggestions"):
                        for suggestion in uncertainty["suggestions"]:
                            st.info(suggestion)
                
            else:
                # Use standard OCR
                extractor = PDFTextExtractor(use_paddle=use_ocr)
                
                if use_ocr:
                    result = extractor.extract_text_from_pdf_with_ocr(temp_path)
                else:
                    result = extractor.extract_text_from_pdf(temp_path)
            
            text = result["full_text"]
            st.session_state.resume_text = text
            
            # Detect language first
            detected_lang = result.get('detected_language', 'en')

            # Phase 1: Layout & ATS Analysis
            layout_detector = LayoutDetector(language=detected_lang)
            layout_features = layout_detector.analyze_layout(text, pdf_path=str(temp_path), lang_code=detected_lang)
            layout_summary = layout_detector.get_layout_summary(text, pdf_path=str(temp_path))

            parser = SectionParser(language=detected_lang)
            parsed = parser.parse(text)
            st.session_state.parsed_resume = parsed
            
            # Initialize extraction results
            aggregated_data = None
            aggregated_skills = []
            unified_data = None
            unified_skills = []
            langextract_data = None
            langextract_skills = []
            
            # Run extraction based on selected method
            if extraction_method == "Aggregated (Best Quality)":
                st.info("🧩 Running aggregated extraction (OCR + Unified + LangExtract where available)...")
                try:
                    aggregated_data = aggregate_resume(Path(temp_path), use_ocr=use_ocr, langextract_passes=2)
                    aggregated_skills = [s.get('name') for s in aggregated_data.get('skills', []) if s.get('name')]
                    st.success(f"✓ Aggregated extraction complete! Found {len(aggregated_data.get('experience', []))} experience items and {len(aggregated_skills)} skills")
                except Exception as e:
                    st.warning(f"⚠ Aggregated extraction failed: {e}")

            if extraction_method == "LangExtract (LLM-Powered)" and LANGEXTRACT_AVAILABLE:
                st.info("🤖 Using LangExtract for AI-powered detailed extraction (this may take 20-60 seconds)...")
                try:
                    if LangExtractResumeParser is None:
                        st.warning("⚠ LangExtract not available in this environment")
                    else:
                        langextract_parser = LangExtractResumeParser()
                        if langextract_parser.is_available():
                            langextract_result = langextract_parser.extract_from_pdf(temp_path, extraction_passes=1)
                            if langextract_result.success:
                                langextract_data = langextract_result.to_dict()
                                # Extract skills from LangExtract data
                                langextract_skills = [s['name'] for s in langextract_data.get('skills', [])]
                                st.success(f"✓ LangExtract extraction complete! Found {len(langextract_data.get('experience', []))} experience items and {len(langextract_skills)} skills")
                            else:
                                st.warning(f"⚠ LangExtract extraction failed: {langextract_result.error_message}")
                        else:
                            st.warning("⚠ LangExtract API key not configured. Set LANGEXTRACT_API_KEY in .env file")
                except Exception as e:
                    st.error(f"❌ LangExtract error: {e}")
                    print(f"LangExtract extraction error: {e}")
            
            # Always run unified extraction as fallback/enhancement
            try:
                unified_extractor = UnifiedResumeExtractor()
                unified_result = unified_extractor.extract(temp_path)
                unified_data = unified_result.to_dict()
                
                # Extract skills from unified data
                unified_skills = []
                for section in unified_data.get('sections', []):
                    if section.get('section_type') == 'skills':
                        for item in section.get('items', []):
                            title = item.get('title', '')
                            desc = item.get('description', '')
                            if title:
                                unified_skills.append(title)
                            if desc:
                                for part in desc.replace('-', ',').split(','):
                                    part = part.strip()
                                    if part:
                                        unified_skills.append(part)
                        break
            except Exception as e:
                unified_data = None
                unified_skills = []
                print(f"Unified extraction error: {e}")
            
            # Phase 3: Enhanced ATS Scoring (with industry-specific weights)
            enhanced_scorer = EnhancedATSScorer(industry=selected_industry.lower())
            
            # Get OCR confidence for enhanced scoring
            ocr_confidence = result.get('overall_confidence', 0.95) if use_ocr else 0.99
            
            # Calculate enhanced score
            enhanced_score = enhanced_scorer.calculate_score(
                text, 
                layout_summary, 
                parsed.to_dict() if hasattr(parsed, 'to_dict') else {
                    "contact_info": parsed.contact_info if hasattr(parsed, 'contact_info') else {},
                    "sections": parsed.sections if hasattr(parsed, 'sections') else [],
                    "skills": parsed.skills if hasattr(parsed, 'skills') else []
                },
                ocr_confidence=ocr_confidence
            )
            enhanced_summary = enhanced_scorer.get_score_summary(enhanced_score)
            
            # Also run standard scorer for comparison
            scorer = ATSScorer()
            parsed_dict = {
                "contact_info": parsed.contact_info,
                "sections": parsed.sections,
                "skills": parsed.skills,
            }
            ats_score = scorer.calculate_score(text, layout_summary, parsed_dict)
            score_summary = scorer.get_score_summary(ats_score)
            
            # Phase 2: Content Analysis
            content_analyzer = ContentAnalyzer()
            content_quality = content_analyzer.analyze(text)
            
            # Phase 3: Content Understanding (Deep Analysis)
            try:
                content_understanding = analyze_resume_content(text)
                content_enrichment = content_understanding.get("enrichment", {})
                content_sections = content_understanding.get("sections", [])
                missing_sections = content_understanding.get("missing_critical_sections", [])
                content_red_flags = content_understanding.get("red_flags", [])
            except Exception as e:
                content_understanding = {}
                content_enrichment = {}
                content_sections = []
                missing_sections = []
                content_red_flags = []
                print(f"Content understanding error: {e}")
            
            # ATS Simulation
            ats_simulator = ATSSimulator()
            ats_simulation = ats_simulator.simulate_parsing(text, layout_summary)
            
            # Store results
            st.session_state.analysis_results = {
                'text': text,
                'layout_summary': layout_summary,
                'layout_features': layout_features,
                'parsed': parsed,
                'ats_score': score_summary,
                'enhanced_ats_score': enhanced_score.to_dict(),
                'enhanced_ats_summary': enhanced_summary,
                'industry': selected_industry.lower(),
                'content_quality': content_quality,
                'content_understanding': content_understanding,
                'content_enrichment': content_enrichment,
                'missing_sections': missing_sections,
                'content_red_flags': content_red_flags,
                'ats_simulation': ats_simulation,
                'detected_language': result.get('detected_language', 'en'),
                'aggregated': aggregated_data,
                'aggregated_skills': aggregated_skills if aggregated_skills else [],
                'unified': unified_data,
                'unified_skills': unified_skills if unified_skills else [],
                'langextract': langextract_data,
                'langextract_skills': langextract_skills if langextract_skills else [],
                'extraction_method': extraction_method,
            }

            # Optional: Persist to Supabase
            if Config.SUPABASE_ENABLED and Config.SUPABASE_AUTO_UPLOAD and SUPABASE_AVAILABLE:
                try:
                    extraction_payload = aggregated_data or langextract_data or unified_data
                    if extraction_payload:
                        store = SupabaseStore()
                        store_result = store.store_resume_and_extraction(
                            Path(temp_path),
                            extraction_payload,
                            extractor=extraction_method,
                            metadata={
                                "detected_language": result.get("detected_language", "en"),
                                "extraction_method": extraction_method,
                                "ocr_used": use_ocr,
                            },
                        )
                        st.caption(f"☁ Stored resume {store_result['resume_id']}")
                    else:
                        st.caption("Supabase storage skipped: no extraction payload available")
                except Exception as e:
                    st.caption(f"Supabase storage failed: {e}")
            
            st.success("Resume processed successfully!")
            
            # Trigger async AI suggestion generation (without blocking spinner)
            if (llm_status['openrouter']['available'] or llm_status['ollama']['available']) and not st.session_state.ai_suggestions:
                st.session_state.ai_suggestions_loading = True
                try:
                    ai_results = generate_ai_suggestions(st.session_state.analysis_results, llm_client)
                    st.session_state.ai_suggestions = ai_results
                except Exception as e:
                    print(f"Error generating AI suggestions: {e}")
                finally:
                    st.session_state.ai_suggestions_loading = False
            
        except Exception as e:
            st.error(f"Error processing resume: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
        finally:
            # Cleanup
            if extractor and not use_ocr:
                extractor.cleanup_temp_images(temp_path)
            if temp_path.exists():
                temp_path.unlink()

# Display results if available
if st.session_state.analysis_results:
    results = st.session_state.analysis_results
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Overview",
        "✍️ Content Quality", 
        "🧠 Content Understanding",
        "💼 Job Matching",
        "🤖 ATS Simulation",
        "💡 Recommendations",
        "📋 Resume Structure",
        "🛠️ Skills Extraction"
    ])
    
    # Tab 1: Overview (Phase 1 features)
    with tab1:
        st.header("ATS Compatibility Score")
        
        score_summary = results['ats_score']
        
        # Score display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("ATS Score", f"{score_summary['overall_score']}/100")
        
        with col2:
            st.metric("Grade", score_summary['grade'])
        
        with col3:
            risk_emoji = {"low": "✅", "medium": "⚠️", "high": "❌"}
            risk = score_summary['risk_level']
            st.metric("Risk Level", f"{risk_emoji.get(risk, '')} {risk.upper()}")
        
        # Score breakdown
        st.subheader("Score Breakdown")
        breakdown = score_summary['breakdown']
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Layout", f"{breakdown['layout']}/25")
        col2.metric("Format", f"{breakdown['format']}/25")
        col3.metric("Content", f"{breakdown['content']}/25")
        col4.metric("Structure", f"{breakdown['structure']}/25")
        
        st.progress(score_summary['overall_score'] / 100)
        
        # Enhanced Scoring Breakdown (Phase 3)
        enhanced_summary = results.get('enhanced_ats_summary', {})
        industry = results.get('industry', 'general')
        
        if enhanced_summary:
            with st.expander("📊 Enhanced ATS Scoring (Phase 3)"):
                st.markdown(f"**Industry Profile: {industry.upper()}**")
                
                enhanced_breakdown = enhanced_summary.get('breakdown', {})
                weighted = enhanced_summary.get('weighted_breakdown', {})
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Parseability", f"{enhanced_breakdown.get('parseability', 0)}/40")
                col2.metric("Structure", f"{enhanced_breakdown.get('structure', 0)}/30")
                col3.metric("Content", f"{enhanced_breakdown.get('content', 0)}/30")
                
                st.markdown("**Industry Weights:**")
                st.caption(f"Parseability: {weighted.get('parseability_weight', 0):.0%} | "
                          f"Structure: {weighted.get('structure_weight', 0):.0%} | "
                          f"Content: {weighted.get('content_weight', 0):.0%}")
                
                # Show issues from enhanced scoring
                enhanced_score = results.get('enhanced_ats_score', {})
                issues = enhanced_score.get('issues', [])
                if issues:
                    st.markdown("**Issues Detected:**")
                    for issue in issues[:5]:
                        severity_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(issue.get('severity', 'low'), "⚪")
                        st.write(f"{severity_emoji} [{issue.get('category', 'general')}] {issue.get('issue', '')}")
        
        st.progress(score_summary['overall_score'] / 100)
        
        # Layout Risk Score (Phase 3)
        layout_features = results.get('layout_features')
        if layout_features:
            with st.expander("📊 Layout Risk Assessment (Phase 3)"):
                try:
                    from parsers.layout_detector import LayoutDetector
                    lang_for_risk = results.get('detected_language', 'en')
                    detector = LayoutDetector(language=lang_for_risk)
                    risk_assessment = detector.calculate_layout_risk_score(layout_features, ocr_confidence=0.95)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Risk Score", f"{risk_assessment['overall_risk_score']:.0f}/100")
                    col2.metric("Risk Level", risk_assessment['risk_level'])
                    col3.metric("ATS Compatibility", risk_assessment['ats_compatibility'])
                    
                    if risk_assessment['factors']:
                        st.write("**Risk Factors:**")
                        for factor in risk_assessment['factors'][:5]:
                            st.write(f"  - {factor['factor']}: {factor['score']:.0f}/{factor['max']}")
                except Exception as e:
                    st.caption(f"Risk assessment unavailable: {e}")
        
        # Detected language
        lang_code = results.get('detected_language', 'en')
        lang_name = LanguageDetector.get_language_name(lang_code)
        st.caption(f"🌐 Detected language: {lang_name}")
        
        # OCR Confidence (Phase 3)
        if ocr_engine_option == "Auto-detect (Recommended)":
            st.caption(f"✓ Enhanced OCR with multi-engine fallback")
        
        # Layout detection method
        layout_features = results.get('layout_features')
        if layout_features:
            detection_method = layout_features.detection_method
            method_emoji = "🤖" if detection_method == "ml" else "📐"
            method_name = "ML (PP-Structure)" if detection_method == "ml" else "Heuristic"
            confidence_text = ""
            if layout_features.confidence is not None:
                confidence_text = f" ({int(layout_features.confidence * 100)}% confidence)"
            st.caption(f"{method_emoji} Layout detection: {method_name}{confidence_text}")
        
        # Content quality score
        st.subheader("Content Quality Score")
        cq = results['content_quality']
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Action Verbs", f"{cq.action_verb_score}/25")
        col2.metric("Quantification", f"{cq.quantification_score}/25")
        col3.metric("Bullet Structure", f"{cq.bullet_structure_score}/25")
        col4.metric("Conciseness", f"{cq.conciseness_score}/25")
        st.progress(cq.overall_score / 100)
        
        # Issues
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⚠️ Issues Detected")
            
            # Get unified data for structure analysis
            unified_data = results.get('unified')
            all_issues = list(score_summary['issues']) if score_summary['issues'] else []
            
            # Add structure-based issues from unified extraction
            if unified_data:
                # Check for missing key sections
                section_types = [s.get('section_type', '') for s in unified_data.get('sections', [])]
                
                if 'summary' not in section_types and 'header' not in section_types:
                    all_issues.append("No professional summary detected - consider adding one")
                
                if 'experience' not in section_types:
                    all_issues.append("No work experience section detected")
                
                if 'education' not in section_types:
                    all_issues.append("No education section detected")
                
                if 'skills' not in section_types:
                    all_issues.append("No skills section detected")
                
                # Check item count issues
                experience_items = sum(1 for s in unified_data.get('sections', []) if s.get('section_type') == 'experience' for _ in s.get('items', []))
                if experience_items == 0:
                    all_issues.append("No experience items could be extracted - check formatting")
                elif experience_items < 2:
                    all_issues.append(f"Only {experience_items} experience item(s) detected - consider adding more")
                
                # Check bullet points
                total_bullets = sum(
                    len(item.get('bullet_points', []))
                    for s in unified_data.get('sections', [])
                    for item in s.get('items', [])
                )
                if total_bullets == 0:
                    all_issues.append("No bullet points detected - use bullets for readability")
            
            if all_issues:
                for issue in all_issues[:10]:
                    st.write(f"- {issue}")
            else:
                st.write("No major issues detected!")
        
        with col2:
            st.subheader("📧 Contact Information")
            
            # Use unified extraction contact if available
            unified_data = results.get('unified')
            if unified_data and unified_data.get('name'):
                st.write(f"**Name:** {unified_data['name']}")
                contact = unified_data.get('contact_info', {})
                st.write(f"**Email:** {contact.get('email', 'Not found')}")
                st.write(f"**Phone:** {contact.get('phone', 'Not found')}")
                st.write(f"**LinkedIn:** {contact.get('linkedin', 'Not found')}")
                st.write(f"**GitHub:** {contact.get('github', 'Not found')}")
            else:
                # Fallback to parsed contact
                contact = results['parsed'].contact_info
                st.write(f"**Name:** {contact.get('name', 'Not found')}")
                st.write(f"**Email:** {contact.get('email', 'Not found')}")
                st.write(f"**Phone:** {contact.get('phone', 'Not found')}")
                st.write(f"**LinkedIn:** {contact.get('linkedin', 'Not found')}")
        
        # Show skills using the new Skill Extractor
        unified_data = results.get('unified')
        if unified_data:
            st.subheader("🛠️ Skills (AI-Extracted)")

            # Detect language from results
            lang = results.get('language', {}).get('detected', 'en')

            # Try to use pre-extracted skills first
            extracted_skills = results.get('extracted_skills', [])

            if not extracted_skills:
                # Use skill extractor as fallback
                from types import SimpleNamespace
                container = SimpleNamespace(sections=unified_data.get('sections', []))
                extracted_skills = extract_skills_from_resume(container, lang)

            if extracted_skills:
                st.success(f"✓ Extracted {len(extracted_skills)} skills")
                # Display skills in a nice format
                cols = st.columns(3)
                for i, skill in enumerate(extracted_skills[:15]):
                    with cols[i % 3]:
                        st.markdown(f"• **{skill}**")
                if len(extracted_skills) > 15:
                    st.caption(f"... and {len(extracted_skills) - 15} more skills")
            else:
                # Fallback to manual extraction
                skills_section = [s for s in unified_data.get('sections', []) if s.get('section_type') == 'skills']
                if skills_section and skills_section[0].get('items'):
                    all_skills = []
                    for item in skills_section[0]['items']:
                        title = item.get('title', '')
                        desc = item.get('description', '')
                        if title:
                            all_skills.append(title)
                        if desc:
                            for part in desc.replace('-', ',').split(','):
                                part = part.strip()
                                if part:
                                    all_skills.append(part)
                    if all_skills:
                        st.write(", ".join(all_skills[:20]))
                        if len(all_skills) > 20:
                            st.caption(f"... and {len(all_skills) - 20} more skills")
                    else:
                        st.warning("No skills detected")
                else:
                    st.warning("No skills section detected")
        
        if show_raw_text:
            st.subheader("📝 Raw Extracted Text")
            st.text_area("Extracted text", results['text'], height=300, label_visibility="collapsed")
    
    # Tab 2: Content Quality
    with tab2:
        st.header("Content Quality Analysis")
        
        cq = results['content_quality']
        
        # Content scores
        st.subheader("Detailed Scores")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Action Verbs", f"{cq.action_verb_score}/25")
            st.caption("Strong verbs like 'Led', 'Developed', 'Increased'")
        with col2:
            st.metric("Quantification", f"{cq.quantification_score}/25")
            st.caption("Metrics like %, $, time saved, users impacted")
        with col3:
            st.metric("Bullet Structure", f"{cq.bullet_structure_score}/25")
            st.caption("Length, consistency, readability")
        with col4:
            st.metric("Conciseness", f"{cq.conciseness_score}/25")
            st.caption("Remove filler words, be direct")
        
        # Action verbs
        st.subheader("🎯 Action Verbs Analysis")
        if cq.action_verbs_found:
            st.write("**Strong verbs found:**")
            st.write(", ".join(cq.action_verbs_found[:15]))
        else:
            st.warning("No strong action verbs detected. Start bullets with verbs like 'Led', 'Developed', 'Created'")
        
        if cq.weak_verbs_found:
            st.write("**Weak verbs to replace:**")
            st.write(", ".join(cq.weak_verbs_found[:10]))
        
        # Quantification
        st.subheader("📊 Quantified Achievements")
        if cq.quantified_achievements:
            st.write(f"Found {len(cq.quantified_achievements)} quantified achievements:")
            for item in cq.quantified_achievements[:5]:
                st.write(f"- {item}")
        else:
            st.warning("No quantified achievements found. Add metrics like 'Increased sales by 25%' or 'Managed $1M budget'")
        
        # Bullet analysis
        st.subheader("📝 Bullet Point Analysis")
        
        # Get bullets from unified extraction
        unified_data = results.get('unified') or {}
        unified_bullets = []
        sections = unified_data.get('sections', []) or []
        
        for section in sections:
            for item in section.get('items', []):
                bullets = item.get('bullet_points', [])
                unified_bullets.extend(bullets)
        
        # Use unified bullets or fallback to CQ bullets
        display_bullets = cq.bullet_points if cq.bullet_points else []
        if unified_bullets:
            st.write(f"**Total bullets (AI-extracted):** {len(unified_bullets)}")
            
            # Show unified bullets by section
            for section in sections[:3]:
                section_bullets = []
                for item in section.get('items', [])[:2]:
                    bullets = item.get('bullet_points', []) or []
                    section_bullets.extend(bullets)
                if section_bullets:
                    with st.expander(f"📂 {section.get('title', 'Section')} ({len(section_bullets)} bullets)"):
                        for bullet in section_bullets[:5]:
                            st.write(f"• {bullet}")
        elif display_bullets:
            st.write(f"**Total bullets analyzed:** {len(display_bullets)}")
        
        # Show sample bullets with issues
        bullets_with_issues = [b for b in display_bullets if b.get('issue')]
        if bullets_with_issues:
            st.write("**Bullets needing improvement:**")
            for bullet in bullets_with_issues[:3]:
                st.info(f"'{bullet['text']}'\n\nIssue: {bullet['issue']}")
        elif not unified_bullets:
            st.warning("No bullet points detected. Use bullets for better readability!")
        
        # Experience items - prefer Aggregated, then LangExtract
        aggregated_data = results.get('aggregated')
        if aggregated_data and aggregated_data.get('experience'):
            st.subheader("💼 Experience Items (Aggregated Extraction)")
            for exp in aggregated_data['experience'][:5]:
                job_title = exp.get('job_title', '')
                company = exp.get('company', '')
                date_range = exp.get('date_range', '')
                location = exp.get('location', '')
                bullet_points = exp.get('bullets', [])
                
                title_display = f"{job_title} at {company}" if job_title and company else (job_title or company or "Experience")
                
                with st.expander(f"📌 {title_display}"):
                    if date_range:
                        st.caption(f"📅 {date_range}")
                    if location:
                        st.caption(f"📍 {location}")
                    if bullet_points:
                        st.write("**Key achievements:**")
                        for bp in bullet_points[:5]:
                            st.write(f"  • {bp}")
        else:
            langextract_data = results.get('langextract')
            if langextract_data and langextract_data.get('experience'):
                st.subheader("💼 Experience Items (LangExtract AI-Extracted)")
                for exp in langextract_data['experience'][:5]:
                    job_title = exp.get('job_title', '')
                    company = exp.get('company', '')
                    date_range = exp.get('date_range', '')
                    location = exp.get('location', '')
                    bullet_points = exp.get('bullet_points', [])
                    
                    title_display = f"{job_title} at {company}" if job_title and company else (job_title or company or "Experience")
                    
                    with st.expander(f"📌 {title_display}"):
                        if date_range:
                            st.caption(f"📅 {date_range}")
                        if location:
                            st.caption(f"📍 {location}")
                        if bullet_points:
                            st.write("**Key achievements:**")
                            for bp in bullet_points[:5]:
                                text = bp.get('text', '')
                                has_metric = bp.get('has_metric', False)
                                if has_metric:
                                    st.write(f"  • {text} 📊")
                                else:
                                    st.write(f"  • {text}")
            else:
                # Fallback to unified extraction
                experience_section = [s for s in sections if s.get('section_type') == 'experience']
                if experience_section:
                    experience_items = experience_section[0].get('items', []) or []
                    if experience_items:
                        st.subheader("💼 Experience Items (AI-Extracted)")
                        for item in experience_items[:5]:
                            title = item.get('title', '') or item.get('subtitle', '')
                            date = item.get('date_range', '')
                            desc = (item.get('description', '') or '')[:150]
                            bullets = item.get('bullet_points', []) or []
                            
                            with st.expander(f"📌 {title}" if title else "Experience Item"):
                                if date:
                                    st.caption(f"📅 {date}")
                                if desc:
                                    st.write(desc + ("..." if len(item.get('description', '') or '') > 150 else ""))
                                if bullets:
                                    st.write("**Key points:**")
                                    for b in bullets[:3]:
                                        st.write(f"  • {b}")
        
        # Recommendations
        if cq.recommendations:
            st.subheader("💡 Content Recommendations")
            for rec in cq.recommendations[:5]:
                st.write(f"- {rec}")
    
    # Tab 3: Job Matching
    with tab3:
        st.header("Job Description Matching")
        
        # Advanced matching options
        col1, col2 = st.columns([1, 1])
        with col1:
            use_advanced_matching = st.checkbox(
                "Use Advanced Matching",
                value=True,
                help="Enable semantic matching, fuzzy matching, and related skills detection"
            )
        with col2:
            show_match_details = st.checkbox(
                "Show Detailed Breakdown",
                value=True,
                help="Display detailed match analysis including match types and confidence scores"
            )
        
        # JD input
        jd_text = st.text_area(
            "Paste job description here",
            height=200,
            placeholder="Paste the full job description text here to see how well your resume matches..."
        )
        
        if jd_text and st.button("Analyze Match"):
            with st.spinner("Analyzing job match with advanced AI..." if use_advanced_matching else "Analyzing job match..."):
                try:
                    # Get skills - try multiple sources in order of quality
                    # 1. Aggregated skills
                    # 2. LangExtract skills (best quality, categorized)
                    # 3. Unified extraction skills
                    # 4. Parsed skills (fallback)
                    skills_to_use = results.get('aggregated_skills', [])
                    if not skills_to_use:
                        skills_to_use = results.get('langextract_skills', [])
                    
                    if not skills_to_use:
                        # Try unified extraction skills
                        skills_to_use = results.get('extracted_skills', [])
                    
                    if not skills_to_use:
                        # Use skill extractor as fallback
                        from types import SimpleNamespace
                        unified_data = results.get('unified')
                        if unified_data:
                            lang = results.get('language', {}).get('detected', 'en')
                            container = SimpleNamespace(sections=unified_data.get('sections', []))
                            skills_to_use = extract_skills_from_resume(container, lang)
                    
                    if not skills_to_use:
                        # Final fallback to parsed skills
                        skills_to_use = results['parsed'].skills
                    
                    if use_advanced_matching:
                        # Use advanced matcher
                        match_result = match_resume_to_job_advanced(
                            results['text'],
                            skills_to_use,
                            jd_text,
                            use_embeddings=use_embeddings
                        )
                    else:
                        # Use basic matcher
                        jd_parser = JobDescriptionParser()
                        jd_data = jd_parser.parse(jd_text)
                        matcher = ResumeJobMatcher(use_embeddings=use_embeddings)
                        match_result = matcher.match(
                            results['text'],
                            skills_to_use,
                            jd_data
                        )
                    
                    # Store for use in other tabs
                    st.session_state.match_result = match_result
                    
                    # Display results
                    st.subheader("Match Results")
                    
                    if use_advanced_matching:
                        # Advanced match display with 4 metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            match_pct = int(match_result.overall_match * 100)
                            st.metric("Overall Match", f"{match_pct}%")
                        with col2:
                            skill_pct = int(match_result.skill_match * 100)
                            st.metric("Skills Match", f"{skill_pct}%")
                        with col3:
                            keyword_pct = int(match_result.keyword_match * 100)
                            st.metric("Keywords Match", f"{keyword_pct}%")
                        with col4:
                            exp_pct = int(match_result.experience_match * 100)
                            st.metric("Experience Match", f"{exp_pct}%")
                    else:
                        # Basic match display
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            match_pct = int(match_result.overall_match * 100)
                            st.metric("Overall Match", f"{match_pct}%")
                        with col2:
                            skill_pct = int(match_result.skill_match * 100)
                            st.metric("Skills Match", f"{skill_pct}%")
                        with col3:
                            keyword_pct = int(match_result.keyword_match * 100)
                            st.metric("Keywords Match", f"{keyword_pct}%")
                    
                    st.progress(match_result.overall_match)
                    
                    # Advanced match details
                    if use_advanced_matching and show_match_details and hasattr(match_result, 'exact_matches'):
                        st.subheader("Match Quality Breakdown")
                        detail_cols = st.columns(4)
                        with detail_cols[0]:
                            st.metric("Exact Matches", match_result.exact_matches)
                        with detail_cols[1]:
                            st.metric("Synonym Matches", match_result.synonym_matches)
                        with detail_cols[2]:
                            st.metric("Fuzzy Matches", match_result.fuzzy_matches)
                        with detail_cols[3]:
                            st.metric("Related Skills", match_result.related_matches)
                    
                    # Skills analysis
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("✅ Matched Skills")
                        if match_result.matched_skills:
                            if use_advanced_matching and hasattr(match_result.matched_skills[0], 'match_type'):
                                # Display with confidence scores
                                for skill_match in match_result.matched_skills[:10]:
                                    confidence = int(skill_match.confidence * 100)
                                    match_type = skill_match.match_type
                                    icon = {
                                        'exact': '✓',
                                        'synonym': '≈',
                                        'fuzzy': '~',
                                        'related': '→'
                                    }.get(match_type, '•')
                                    
                                    exp_info = ""
                                    if skill_match.experience_years:
                                        exp_info = f" ({skill_match.experience_years:.1f} yrs)"
                                    
                                    st.success(f"{icon} {skill_match.skill_name} ({confidence}%)" + exp_info)
                            else:
                                # Basic display
                                for skill in match_result.matched_skills[:10]:
                                    st.success(skill)
                        else:
                            st.write("No matching skills found")
                    
                    with col2:
                        st.subheader("❌ Missing Skills")
                        if match_result.missing_skills:
                            for skill in match_result.missing_skills[:10]:
                                st.error(skill)
                        elif match_result.matched_skills:
                            st.write("No missing skills - great match!")
                        else:
                            st.write("No skills found in job description to match")
                    
                    # Related skills (advanced only)
                    if use_advanced_matching and hasattr(match_result, 'related_skills') and match_result.related_skills and show_match_details:
                        st.subheader("🔗 Related Skills (Partial Credit)")
                        st.write("These skills from your resume are related to job requirements:")
                        for related in match_result.related_skills[:5]:
                            confidence = int(related.confidence * 100)
                            st.info(f"{related.skill_name} ({confidence}%) - {related.context or 'Related skill'}")
                    
                    # Keywords
                    if match_result.missing_keywords:
                        st.subheader("Missing Keywords to Add")
                        st.write(", ".join(match_result.missing_keywords[:15]))
                    
                    # Recommendations
                    if match_result.recommendations:
                        st.subheader("Match Recommendations")
                        for rec in match_result.recommendations:
                            st.write(f"- {rec}")
                    
                except Exception as e:
                    st.error(f"Error analyzing job match: {e}")
        elif not jd_text:
            st.info("👆 Paste a job description above to see how well your resume matches")
    
    # Tab 4: ATS Simulation
    with tab4:
        st.header("ATS Parser Simulation")
        
        sim = results['ats_simulation']
        unified_data = results.get('unified')
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Parsing Confidence", f"{int(sim.parsing_confidence * 100)}%")
        with col2:
            st.metric("Readability Score", f"{int(sim.readability_score * 100)}%")
        
        # What ATS sees
        st.subheader("👁️ What ATS Systems See")
        st.text_area("Parsed content", sim.plain_text, height=250, label_visibility="collapsed")
        
        # Sections detected - USE UNIFIED EXTRACTION
        st.subheader("📑 Detected Sections")
        
        unified_sections = []
        if unified_data:
            unified_sections = unified_data.get('sections', [])
        
        if unified_sections:
            st.success(f"AI detected {len(unified_sections)} sections")
            for section in unified_sections:
                section_title = section.get('title', 'Unknown')
                section_type = section.get('section_type', 'unknown')
                items_count = len(section.get('items', []))
                raw_text = section.get('raw_text', '')[:500]
                
                with st.expander(f"{section_title} ({section_type}) - {items_count} items"):
                    st.write(f"**Type:** {section_type}")
                    st.write(f"**Items:** {items_count}")
                    
                    # Show items
                    for i, item in enumerate(section.get('items', [])[:5]):
                        item_title = item.get('title', '') or item.get('subtitle', '')
                        item_date = item.get('date_range', '')
                        item_desc = item.get('description', '')[:200]
                        
                        st.markdown(f"**{i+1}. {item_title}**" if item_title else f"**Item {i+1}**")
                        if item_date:
                            st.caption(f"📅 {item_date}")
                        if item_desc:
                            st.write(item_desc + ("..." if len(item.get('description', '')) > 200 else ""))
                    
                    if len(section.get('items', [])) > 5:
                        st.caption(f"... and {len(section.get('items', [])) - 5} more items")
                    
                    # Raw text
                    if raw_text:
                        st.divider()
                        st.caption("Raw section text:")
                        st.text(raw_text)
        elif sim.extracted_sections:
            # Fallback to ATS simulation sections
            st.warning("Using fallback section detection")
            for section, content in sim.extracted_sections.items():
                with st.expander(f"{section.title()} ({len(content)} chars)"):
                    st.text(content[:500] + "..." if len(content) > 500 else content)
        else:
            st.warning("No clear sections detected")
        
        # Skills detected - USE NEW SKILL EXTRACTOR
        st.subheader("🛠️ Skills Detected")
        
        # Show which extraction method was used
        extraction_method = results.get('extraction_method', 'Standard')
        if extraction_method == "Aggregated (Best Quality)":
            st.info("📊 Showing skills extracted using aggregated pipeline")
        elif extraction_method == "LangExtract (LLM-Powered)":
            st.info("📊 Showing skills extracted using LangExtract (AI-powered categorization)")
        
        # Check for Aggregated skills first (best quality)
        aggregated_data = results.get('aggregated')
        aggregated_skills = results.get('aggregated_skills', [])

        if aggregated_data and aggregated_skills:
            st.success(f"✓ Extracted {len(aggregated_skills)} skills using aggregated pipeline")
            skills_by_category = {}
            for skill in aggregated_data.get('skills', []):
                category = skill.get('category', 'Other')
                if category not in skills_by_category:
                    skills_by_category[category] = []
                skills_by_category[category].append(skill.get('name', ''))
            for category, skills in sorted(skills_by_category.items()):
                if skills:
                    with st.expander(f"📁 {category.replace('_', ' ').title()} ({len(skills)} skills)"):
                        cols = st.columns(3)
                        for i, skill in enumerate(skills):
                            cols[i % 3].write(f"• {skill}")
        else:
            # Check for LangExtract skills
            langextract_data = results.get('langextract')
            langextract_skills = results.get('langextract_skills', [])

            if langextract_data and langextract_skills:
                # Display LangExtract skills with categorization
                st.success(f"✓ Extracted {len(langextract_skills)} skills using AI categorization")

                # Group skills by category
                skills_by_category = {}
                for skill in langextract_data.get('skills', []):
                    category = skill.get('category', 'Other')
                    if category not in skills_by_category:
                        skills_by_category[category] = []
                    skills_by_category[category].append(skill.get('name', ''))

                # Display by category
                for category, skills in sorted(skills_by_category.items()):
                    if skills:
                        with st.expander(f"📁 {category.replace('_', ' ').title()} ({len(skills)} skills)"):
                            cols = st.columns(3)
                            for i, skill in enumerate(skills):
                                cols[i % 3].write(f"• {skill}")
            else:
                # Fallback to unified skills
                lang = results.get('language', {}).get('detected', 'en')
                unified_skills = []

                if unified_data:
                    # Use the new extract_skills_from_resume function
                    unified_skills = results.get('extracted_skills', [])
                    if not unified_skills:
                        # Fallback: use types.SimpleNamespace
                        from types import SimpleNamespace
                        container = SimpleNamespace(sections=unified_data.get('sections', []))
                        unified_skills = extract_skills_from_resume(container, lang)

                if unified_skills:
                    st.success(f"✓ Extracted {len(unified_skills)} skills")
                    cols = st.columns(3)
                    for i, skill in enumerate(unified_skills):
                        with cols[i % 3]:
                            st.markdown(f"• **{skill}**")
                elif sim.detected_skills:
                    st.warning("Using fallback skill detection")
                    st.write(", ".join(sim.detected_skills))
                else:
                    st.warning("No skills detected")
        
        # Warnings
        if sim.warnings:
            st.subheader("⚠️ ATS Warnings")
            for warning in sim.warnings:
                st.warning(warning)
        
        # Lost content
        if sim.lost_content:
            st.subheader("🚨 Potentially Lost Content")
            for item in sim.lost_content:
                st.error(item)
    
    # Tab 5: Recommendations
    with tab5:
        st.header("Prioritized Recommendations")
        
        # Generate recommendations
        rec_engine = RecommendationEngine(llm_client=llm_client)
        
        # Get job match if available
        job_match_data = None
        if 'match_result' in st.session_state and st.session_state.match_result:
            match_result = st.session_state.match_result
            job_match_data = {
                'overall_match': match_result.overall_match,
                'missing_skills': match_result.missing_skills,
                'missing_keywords': match_result.missing_keywords,
            }
        
        # Get unified extraction data for structure analysis
        unified_data = results.get('unified', {})
        structure_analysis = {
            'sections_detected': len(unified_data.get('sections', [])),
            'has_summary': bool(unified_data.get('summary')),
            'has_contact': bool(unified_data.get('contact_info')),
            'total_items': sum(len(s.get('items', [])) for s in unified_data.get('sections', [])),
        }
        
        recommendations = rec_engine.generate_recommendations(
            results['ats_score'],
            {
                'action_verb_score': results['content_quality'].action_verb_score,
                'quantification_score': results['content_quality'].quantification_score,
                'bullet_structure_score': results['content_quality'].bullet_structure_score,
                'conciseness_score': results['content_quality'].conciseness_score,
                'weak_verbs_found': results['content_quality'].weak_verbs_found,
                'structure_analysis': structure_analysis,
            },
            results['layout_summary'],
            job_match_data
        )
        
        # Priority summary
        summary = rec_engine.get_priority_summary(recommendations)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("🔴 High Priority", summary['high'])
        col2.metric("🟡 Medium Priority", summary['medium'])
        col3.metric("🟢 Low Priority", summary['low'])
        
        # Display recommendations by priority
        for priority in [Priority.HIGH, Priority.MEDIUM, Priority.LOW]:
            priority_recs = [r for r in recommendations if r.priority == priority]
            
            if priority_recs:
                emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}[priority.value]
                st.subheader(f"{emoji} {priority.value.title()} Priority")
                
                for rec in priority_recs[:5]:  # Show top 5 per priority
                    with st.expander(f"{rec.category}: {rec.issue}"):
                        st.write(f"**Suggestion:** {rec.suggestion}")
                        if rec.example:
                            st.write(f"**Example:** {rec.example}")
                        if rec.estimated_impact:
                            st.write(f"**Estimated Impact:** {rec.estimated_impact}")
        
        # AI-powered suggestions
        st.subheader("🤖 AI-Powered Improvements")
        
        # Display cached AI suggestions or loading state
        if st.session_state.ai_suggestions_loading:
            st.info("⏳ AI suggestions are being generated... Check back in a moment!")
        elif st.session_state.ai_suggestions:
            ai_suggestions = st.session_state.ai_suggestions
            suggestions_generated = False
            
            # Display keywords
            if ai_suggestions.get('keywords'):
                st.write("**🎯 Suggested Keywords to Add:**")
                st.write(ai_suggestions['keywords'])
                suggestions_generated = True
            
            # Display improved bullets
            if ai_suggestions.get('improved_bullets'):
                st.write("**✍️ Improved Bullet Points:**")
                for bullet in ai_suggestions['improved_bullets']:
                    st.info(f"**Before:** {bullet['before']}\n\n**After:** {bullet['after']}")
                    suggestions_generated = True
            
            # Display enhancements for high-scoring resumes
            if ai_suggestions.get('enhancements'):
                st.write("**🚀 Enhancement Opportunities (AI Critique):**")
                st.caption("Even strong bullets can be optimized for maximum impact")
                for enhancement in ai_suggestions['enhancements']:
                    st.info(f"**Current:** {enhancement['current']}\n\n**💡 Enhancement:** {enhancement['enhancement']}")
                    suggestions_generated = True
            
            if not suggestions_generated:
                st.info("✅ No AI improvements needed - your resume looks great!")
        elif not (llm_status['openrouter']['available'] or llm_status['ollama']['available']):
            st.error("No LLM available. Set OPENROUTER_API_KEY or install Ollama for AI suggestions.")
    
    # Tab 6: Resume Structure (Unified Extraction)
    with tab6:
        st.header("📋 Resume Structure (AI-Parsed)")
        
        unified_data = results.get('unified')
        
        if unified_data:
            # Name and Contact
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("👤 Name")
                st.write(unified_data.get('name', 'Not detected'))
            with col2:
                st.subheader("📧 Contact")
                contact = unified_data.get('contact_info', {})
                if contact:
                    st.write(f"**Email:** {contact.get('email', 'N/A')}")
                    st.write(f"**Phone:** {contact.get('phone', 'N/A')}")
                    st.write(f"**LinkedIn:** {contact.get('linkedin', 'N/A')}")
                    st.write(f"**GitHub:** {contact.get('github', 'N/A')}")
            
            # Summary
            if unified_data.get('summary'):
                st.subheader("📝 Professional Summary")
                st.info(unified_data['summary'])
            
            # Sections with items
            st.subheader("📑 Resume Sections")
            
            for section in unified_data.get('sections', []):
                if section.get('items'):
                    with st.expander(f"{section['title']} ({section.get('section_type', 'unknown')}) - {len(section['items'])} items"):
                        for item in section['items']:
                            # Item header
                            item_title = item.get('title', '') or item.get('subtitle', '')
                            item_date = item.get('date_range', '')
                            item_location = item.get('location', '')
                            item_company = item.get('company', '')
                            
                            # Build header
                            header_parts = []
                            if item_title:
                                header_parts.append(f"**{item_title}**")
                            if item_company:
                                header_parts.append(f"@ {item_company}")
                            if item_location:
                                header_parts.append(f"📍 {item_location}")
                            if item_date:
                                header_parts.append(f"📅 {item_date}")
                            
                            if header_parts:
                                st.write(" | ".join(header_parts))
                            
                            # Description
                            description = item.get('description', '')
                            if description:
                                st.write(description[:300] + "..." if len(description) > 300 else description)
                            
                            # Bullet points
                            bullets = item.get('bullet_points', [])
                            if bullets:
                                for bullet in bullets[:5]:
                                    st.write(f"  • {bullet}")
                                if len(bullets) > 5:
                                    st.caption(f"... and {len(bullets) - 5} more")
                            
                            st.divider()
            
            # Skills from unified extraction
            skills_section = [s for s in unified_data.get('sections', []) if s.get('section_type') == 'skills']
            if skills_section and skills_section[0].get('items'):
                st.subheader("🛠️ Technical Skills (AI-Extracted)")
                all_skills = []
                for item in skills_section[0]['items']:
                    title = item.get('title', '')
                    desc = item.get('description', '')
                    if title and title not in all_skills:
                        all_skills.append(title)
                    if desc:
                        for part in desc.replace('-', ',').split(','):
                            part = part.strip()
                            if part and part not in all_skills:
                                all_skills.append(part)
                if all_skills:
                    st.write(", ".join(all_skills))
            
            # Full extracted text
            with st.expander("📄 Full Extracted Text"):
                st.text(unified_data.get('all_text', 'N/A')[:5000])
                if len(unified_data.get('all_text', '')) > 5000:
                    st.caption(f"... and {len(unified_data.get('all_text', '')) - 5000} more characters")
        
        else:
            st.warning("Unified extraction failed. Using fallback extraction.")
            # Fallback to standard parsed data
            parsed = results.get('parsed')
            if parsed:
                st.subheader("📑 Sections (Fallback)")
                for section in parsed.sections:
                    st.write(f"**{section['name']}** ({section.get('section_type', 'unknown')})")
                    st.write(section.get('content', '')[:200] + "..." if len(section.get('content', '')) > 200 else section.get('content', ''))
                    st.divider()
        
        # Enhanced job matching with unified data
        st.divider()
        st.subheader("🔗 Enhanced Job Matching")
        st.write("Unified extraction provides better structure for job matching:")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.success("Clean section titles")
        with col2:
            st.success("Grouped experience items")
        with col3:
            st.success("Parsed dates & locations")
        
        st.info("Job matching in the **Job Matching** tab now uses this structured data!")
    
    # Tab 7: Content Understanding (Phase 3)
    with tab7:
        st.header("Content Understanding (Phase 3)")
        
        content_understanding = results.get('content_understanding', {})
        content_enrichment = results.get('content_enrichment', {})
        missing_sections = results.get('missing_sections', [])
        content_red_flags = results.get('content_red_flags', [])
        
        # Section Detection
        st.subheader("Detected Sections")
        sections = content_understanding.get('sections', [])
        if sections:
            for section in sections:
                section_type = section.get('type', 'unknown')
                title = section.get('title', 'Unknown')
                confidence = section.get('confidence', 0)
                items = section.get('item_count', 0)
                
                st.write(f"**{title}** ({section_type}) - Items: {items}, Confidence: {confidence:.0%}")
        else:
            st.info("No sections detected")
        
        # Missing Sections
        if missing_sections:
            st.subheader("Missing Critical Sections")
            for section in missing_sections:
                st.error(f"Missing: {section}")
        
        # Content Enrichment
        st.subheader("Content Enrichment Insights")
        
        enrichment = results.get('content_enrichment', {})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            seniority = enrichment.get('estimated_seniority', 'Unknown')
            st.metric("Estimated Seniority", seniority.title() if seniority else "Unknown")
        with col2:
            exp_years = enrichment.get('total_experience_years', 0)
            st.metric("Total Experience", f"{exp_years:.0f} years")
        with col3:
            themes = enrichment.get('key_themes', [])
            st.metric("Key Themes", len(themes))
        
        # Key Themes
        themes = enrichment.get('key_themes', [])
        if themes:
            st.subheader("Key Themes")
            for theme in themes:
                st.info(theme)
        
        # Red Flags
        if content_red_flags:
            st.subheader("Red Flags")
            for flag in content_red_flags:
                severity = flag.get('severity', 'low')
                category = flag.get('category', 'general')
                description = flag.get('description', '')
                suggestion = flag.get('suggestion', '')
                
                with st.expander(f"[{severity.upper()}] {category}: {description}"):
                    st.write(f"Suggestion: {suggestion}")
    
    # Tab 8: Skills Extraction (Phase 4)
    with tab8:
        st.header("Skills Extraction (Phase 4)")
        
        skills_extraction = results.get('skills_extraction', {})
        skills_data = skills_extraction.get('skills', [])
        
        if skills_data:
            st.subheader(f"Extracted Skills ({len(skills_data)} total)")
            
            # Group skills by category
            skills_by_category = {}
            for skill in skills_data:
                cat = skill.get('category', 'unknown')
                if cat not in skills_by_category:
                    skills_by_category[cat] = []
                skills_by_category[cat].append(skill)
            
            # Display by category
            for category, category_skills in skills_by_category.items():
                with st.expander(f"{category.replace('_', ' ').title()} ({len(category_skills)} skills)"):
                    for skill in category_skills:
                        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                        with col1:
                            st.write(f"**{skill.get('name', skill.get('canonical_name', 'Unknown'))}**")
                        with col2:
                            confidence = skill.get('confidence', 0)
                            st.write(f"Conf: {confidence:.0%}")
                        with col3:
                            proficiency = skill.get('proficiency', 'unknown')
                            st.write(f"Level: {proficiency}")
                        with col4:
                            is_explicit = skill.get('is_explicit', True)
                            emoji = "Explicit" if is_explicit else "Implicit"
                            st.write(f"{emoji}")
            
            # Category breakdown
            st.subheader("Skills by Category")
            categories = skills_extraction.get('categories_found', [])
            col_list = st.columns(min(len(categories), 4))
            for i, cat in enumerate(categories):
                with col_list[i % 4]:
                    count = len([s for s in skills_data if s.get('category') == cat])
                    st.metric(cat.replace('_', ' ').title(), count)
            
            # Related skills
            st.subheader("Skill Relationships")
            st.info("Skills that are related or complementary to each other")
            
        else:
            st.info("No skills extracted. Upload a resume with skill information.")

else:
    st.info("Upload a PDF resume to get started")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""**ATS Resume Analyzer**
Phase 4 Enhanced""")
