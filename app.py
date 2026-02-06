"""Streamlit app for ATS Resume Analyzer - Phase 2 Enhanced."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
from parsers import PDFTextExtractor, LayoutDetector, SectionParser
from parsers.language_detector import LanguageDetector
from scoring import ATSScorer
from analysis import (
    ContentAnalyzer,
    JobDescriptionParser,
    ResumeJobMatcher,
    RecommendationEngine,
    ATSSimulator,
    LLMClient,
    Priority,
)
from utils import Config

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

# LLM Status
st.sidebar.markdown("---")
st.sidebar.markdown("**LLM Available**")

if llm_status['openrouter']['available']:
    st.sidebar.markdown("✓ OpenRouter")

if llm_status['ollama']['available']:
    st.sidebar.markdown("✓ Ollama")
    
if not llm_status['openrouter']['available'] and not llm_status['ollama']['available']:
    st.sidebar.caption("Optional: Configure for AI features")

# File upload
st.header("Upload Resume")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_path = Config.PROCESSED_DIR / uploaded_file.name
    Config.PROCESSED_DIR.mkdir(exist_ok=True)
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    # Process the resume
    extractor = None
    with st.spinner("Processing resume..."):
        try:
            # Extract text
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
            layout_features = layout_detector.analyze_layout(text, lang_code=detected_lang)
            layout_summary = layout_detector.get_layout_summary(text)

            parser = SectionParser(language=detected_lang)
            parsed = parser.parse(text)
            st.session_state.parsed_resume = parsed
            
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
            
            # ATS Simulation
            ats_simulator = ATSSimulator()
            ats_simulation = ats_simulator.simulate_parsing(text, layout_summary)
            
            # Store results
            st.session_state.analysis_results = {
                'text': text,
                'layout_summary': layout_summary,
                'parsed': parsed,
                'ats_score': score_summary,
                'content_quality': content_quality,
                'ats_simulation': ats_simulation,
                'detected_language': result.get('detected_language', 'en'),
            }
            
            st.success("Resume processed successfully!")
            
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "✍️ Content Quality", 
        "💼 Job Matching",
        "🤖 ATS Simulation",
        "💡 Recommendations"
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
        
        # Detected language
        lang_code = results.get('detected_language', 'en')
        lang_name = LanguageDetector.get_language_name(lang_code)
        st.caption(f"🌐 Detected language: {lang_name}")
        
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
            if score_summary['issues']:
                for issue in score_summary['issues']:
                    st.write(f"- {issue}")
            else:
                st.write("No major issues detected!")
        
        with col2:
            st.subheader("📧 Contact Information")
            contact = results['parsed'].contact_info
            st.write(f"**Name:** {contact.get('name', 'Not found')}")
            st.write(f"**Email:** {contact.get('email', 'Not found')}")
            st.write(f"**Phone:** {contact.get('phone', 'Not found')}")
            st.write(f"**LinkedIn:** {contact.get('linkedin', 'Not found')}")
        
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
        if cq.bullet_points:
            st.write(f"**Total bullets analyzed:** {len(cq.bullet_points)}")
            
            # Show sample bullets with issues
            bullets_with_issues = [b for b in cq.bullet_points if b.get('issue')]
            if bullets_with_issues:
                st.write("**Bullets needing improvement:**")
                for bullet in bullets_with_issues[:3]:
                    st.info(f"'{bullet['text']}'\n\nIssue: {bullet['issue']}")
        
        # Recommendations
        if cq.recommendations:
            st.subheader("💡 Content Recommendations")
            for rec in cq.recommendations[:5]:
                st.write(f"- {rec}")
    
    # Tab 3: Job Matching
    with tab3:
        st.header("Job Description Matching")
        
        # JD input
        jd_text = st.text_area(
            "Paste job description here",
            height=200,
            placeholder="Paste the full job description text here to see how well your resume matches..."
        )
        
        if jd_text and st.button("Analyze Match"):
            with st.spinner("Analyzing job match..."):
                try:
                    # Parse JD
                    jd_parser = JobDescriptionParser()
                    jd_data = jd_parser.parse(jd_text)
                    
                    # Match
                    matcher = ResumeJobMatcher(use_embeddings=use_embeddings)
                    match_result = matcher.match(
                        results['text'],
                        results['parsed'].skills,
                        jd_data
                    )
                    
                    # Store for use in other tabs
                    st.session_state.match_result = match_result
                    
                    # Display results
                    st.subheader("Match Results")
                    
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
                    
                    # Skills analysis
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("✅ Matched Skills")
                        if match_result.matched_skills:
                            for skill in match_result.matched_skills[:10]:
                                st.success(skill)
                        else:
                            st.write("No matching skills found")
                    
                    with col2:
                        st.subheader("❌ Missing Skills")
                        if match_result.missing_skills:
                            for skill in match_result.missing_skills[:10]:
                                st.error(skill)
                        else:
                            st.write("No missing skills - great match!")
                    
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
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Parsing Confidence", f"{int(sim.parsing_confidence * 100)}%")
        with col2:
            st.metric("Readability Score", f"{int(sim.readability_score * 100)}%")
        
        # What ATS sees
        st.subheader("👁️ What ATS Systems See")
        st.text_area("Parsed content", sim.plain_text, height=250, label_visibility="collapsed")
        
        # Sections detected
        st.subheader("📑 Detected Sections")
        if sim.extracted_sections:
            for section, content in sim.extracted_sections.items():
                with st.expander(f"{section.title()} ({len(content)} chars)"):
                    st.text(content[:500] + "..." if len(content) > 500 else content)
        else:
            st.warning("No clear sections detected")
        
        # Skills detected
        st.subheader("🛠️ Skills Detected by ATS")
        if sim.detected_skills:
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
        
        recommendations = rec_engine.generate_recommendations(
            results['ats_score'],
            {
                'action_verb_score': results['content_quality'].action_verb_score,
                'quantification_score': results['content_quality'].quantification_score,
                'bullet_structure_score': results['content_quality'].bullet_structure_score,
                'conciseness_score': results['content_quality'].conciseness_score,
                'weak_verbs_found': results['content_quality'].weak_verbs_found,
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
        
        if st.button("Generate AI Suggestions"):
            print(f"LLM Client: {llm_client}")
            print(f"LLM Status: {llm_status}")
            
            if llm_client:
                print(f"OpenRouter available: {llm_client.openrouter_available}")
                print(f"Ollama available: {llm_client.ollama_available}")
            
            if (llm_status['openrouter']['available'] or llm_status['ollama']['available']):
                with st.spinner("Generating AI suggestions..."):
                    try:
                        suggestions_generated = False
                        
                        # Suggest keywords
                        print(f"Calling suggest_keywords with resume text length: {len(results['text'])}")
                        keyword_response = llm_client.suggest_keywords(results['text'])
                        print(f"Keywords response: success={keyword_response.success}, error={keyword_response.error}, text_length={len(keyword_response.text) if keyword_response.text else 0}")
                        
                        if keyword_response.success and keyword_response.text.strip():
                            st.write("**🎯 Suggested Keywords to Add:**")
                            st.write(keyword_response.text)
                            suggestions_generated = True
                        elif keyword_response.error:
                            st.warning(f"Keywords failed: {keyword_response.error}")
                        
                        # Improve weak bullets
                        weak_bullets = [b for b in results['content_quality'].bullet_points if b.get('issue')]
                        if weak_bullets:
                            st.write("**✍️ Improved Bullet Points:**")
                            for bullet in weak_bullets[:3]:
                                improved = llm_client.improve_bullet_point(bullet['text'])
                                if improved.success and improved.text.strip():
                                    st.info(f"**Before:** {bullet['text']}\n\n**After:** {improved.text}")
                                    suggestions_generated = True
                                elif improved.error:
                                    st.warning(f"Bullet improvement failed: {improved.error}")
                        else:
                            st.write("No bullet points need improvement - your resume looks great!")
                        
                        if not suggestions_generated and weak_bullets:
                            st.warning("No AI suggestions were generated. The LLM service may be unavailable.")
                    
                    except Exception as e:
                        st.error(f"Error generating AI suggestions: {e}")
                        import traceback
                        st.error(traceback.format_exc())
            else:
                st.error("No LLM available. Set OPENROUTER_API_KEY or install Ollama.")

else:
    st.info("👆 Upload a PDF resume to get started")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""**ATS Resume Analyzer**
Phase 2 Enhanced""")
