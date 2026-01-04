Multimodal Fake News Forensics Project

This project implements a modular fake news detection system that analyzes textual content and visual information to identify misinformation. The system is designed with a clean separation between backend inference, frontend interaction, and model pipelines, enabling easy scalability and future multimodal integration.

Currently, the project includes a fully functional text fake news detection pipeline with explainable predictions and a dataset-independent image forensics inference pipeline using pretrained models. A Streamlit-based frontend allows users to interactively test news samples, while the backend APIs handle model inference.

The architecture supports seamless extension to multimodal fusion, where text and image signals can be combined once image datasets are integrated.

Here are a few samples for text news:
1. Real news:
    Sample 1: Health authorities stated that no verified scientific studies have established a direct link between packaged milk consumption and cancer. Experts emphasized the importance of relying on peer-reviewed research rather than unverified online claims.

    Sample 2: The Ministry of Health announced updated food safety guidelines following routine inspections and laboratory testing conducted across multiple regions. Officials clarified that the changes were part of standard regulatory procedures.

2. Fake news:
    Sample 3: Secret reports allegedly prove that packaged milk contains hidden chemicals that cause cancer, but government agencies are suppressing this information to prevent public outrage.

    Sample 4: Anonymous insiders revealed that food safety authorities knowingly approved harmful products after being pressured by large corporations, according to unverified documents circulating online.

3. Uncertain:
    Sample 5: Recent discussions on social media suggest that long-term consumption of certain processed foods may have unknown health effects, although no official confirmation has been provided.
    
    Sample 6: Some commentators claim that changes in food regulations could impact consumer health, but experts note that more evidence is required before drawing conclusions.