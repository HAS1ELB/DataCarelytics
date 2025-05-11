import streamlit as st
import os
from medical_report_analysis.agents import Cardiologist, Psychologist, Pulmonologist, MultidisciplinaryTeam

def main():
    st.markdown("<h2 class='sub-header'>Medical Report Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<div class='category-box'>", unsafe_allow_html=True)

    # File upload section
    st.subheader("Upload Medical Report")
    uploaded_file = st.file_uploader("Choose a medical report (.txt)", type=["txt"])

    if uploaded_file:
        try:
            # Read the medical report
            medical_report = uploaded_file.read().decode("utf-8")
            st.success("✅ Medical report uploaded successfully!")

            # Process the report with specialist agents
            with st.spinner("Analyzing report with specialist agents..."):
                agents = {
                    "Cardiologist": Cardiologist(medical_report),
                    "Psychologist": Psychologist(medical_report),
                    "Pulmonologist": Pulmonologist(medical_report)
                }

                responses = {}
                for name, agent in agents.items():
                    st.write(f"Running {name} analysis...")
                    responses[name] = agent.run()

                # Run multidisciplinary team analysis
                st.write("Generating final diagnosis...")
                team_agent = MultidisciplinaryTeam(
                    cardiologist_report=responses["Cardiologist"],
                    psychologist_report=responses["Psychologist"],
                    pulmonologist_report=responses["Pulmonologist"]
                )
                final_diagnosis = team_agent.run()

                # Display final diagnosis
                st.markdown("### Final Diagnosis")
                st.markdown("<div class='results-container'>", unsafe_allow_html=True)
                for line in final_diagnosis.split('\n'):
                    if line.startswith('-'):
                        st.markdown(f"• {line[1:]}")
                    elif ':' in line:
                        parts = line.split(':', 1)
                        st.markdown(f"**{parts[0]}**: {parts[1]}")
                    else:
                        st.markdown(line)
                st.markdown("</div>", unsafe_allow_html=True)

                '''# Save diagnosis to file
                os.makedirs("results", exist_ok=True)
                with open("medical_report_analysis/results/final_diagnosis.txt", "w") as f:
                    f.write(f"### Final Diagnosis:\n\n{final_diagnosis}")
                st.info("Diagnosis saved to results/final_diagnosis.txt")
'''
        except Exception as e:
            st.error(f"Error processing report: {str(e)}")

    else:
        st.info("Please upload a .txt medical report to begin analysis.")

    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()