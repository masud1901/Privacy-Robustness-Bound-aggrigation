# Literature Review on Privacy-Robustness Trade-off in Federated Learning

Federated learning (FL) enables collaborative model training across distributed clients without sharing raw data, making it a promising approach for privacy-sensitive applications such as healthcare, finance, and IoT. However, FL systems face significant challenges in simultaneously ensuring privacy against inference attacks and robustness against adversarial attacks, such as poisoning. The privacy-robustness trade-off is a critical research area, as privacy-preserving mechanisms like differential privacy (DP) often degrade model performance or increase vulnerability to attacks, while robustness-enhancing techniques may risk privacy leakage. This literature review synthesizes key findings from recent surveys and research papers, focusing on theoretical frameworks, practical methods, and future directions for addressing this trade-off. The review is structured around a comprehensive matrix of relevant literature, expanded to include additional papers as requested, to provide a broad and detailed perspective on the topic.

## Background and Significance

FL operates by having clients train local models on private data and send updates (e.g., gradients) to a central server for aggregation, forming a global model. This decentralized approach reduces data exposure compared to centralized machine learning but introduces vulnerabilities. Privacy attacks, such as membership inference or model inversion, can extract sensitive information from model updates, while robustness threats, like Byzantine attacks, involve malicious clients sending poisoned updates to degrade model performance or inject backdoors. The privacy-robustness trade-off arises because privacy mechanisms, such as DP, add noise that can obscure legitimate updates, making it harder to detect malicious ones, while robust aggregation methods, like outlier filtering, may require analyzing updates in ways that risk privacy leakage.

The user-provided context introduces a novel theoretical framework, the Privacy-Robustness Bound (PRB), defined as PRB(ε, τ, δ) = C₁·(ε + 1/ε) + C₂·τ·√(d/n) + C₃·√(log(1/δ)/n), where ε is the privacy parameter, τ is the fraction of malicious clients, d is the model dimension, n is the number of clients, and δ is the probability of guarantee violation. PRB quantifies the trade-off by deriving an optimal privacy budget (ε\* = √(C₁/C₂·τ·√(d/n))) that minimizes model deviation under poisoning attacks, validated through experiments on datasets like CIFAR-10 and FEMNIST. This review builds on such contributions by exploring how the broader literature addresses this trade-off, emphasizing theoretical insights, practical solutions, and comprehensive surveys.

## Key Themes in the Literature

### Theoretical Frameworks for the Privacy-Robustness Trade-off

Theoretical studies provide a foundation for understanding the privacy-robustness trade-off. A pivotal work, "Understanding the Interplay between Privacy and Robustness in Federated Learning" (2021) (arXiv), investigates how local differential privacy (LDP) affects adversarial robustness in FL. Through theoretical analysis and empirical experiments, the authors demonstrate that LDP, which adds noise to client updates, can have dual effects: it may enhance robustness by masking malicious updates but can also degrade robustness by introducing excessive noise that obscures legitimate updates. This duality underscores the need for careful parameter tuning to balance privacy and robustness.

The PRB framework from the user’s context extends this by formalizing the trade-off with a unified metric, showing that model deviation under attacks is bounded by PRB(ε, τ, δ) with high probability (&gt;95%). The framework’s optimal privacy budget (ε\*) scales with the fraction of malicious clients and model complexity, providing a rigorous basis for designing FL systems. Similarly, "Federated Learning with Local Differential Privacy: Trade-offs between Privacy, Utility, and Communication" (2021) (arXiv) derives theoretical bounds on how LDP impacts utility and communication, indirectly relating to robustness by showing how privacy affects model performance, which can influence vulnerability to attacks.

### Privacy-Preserving Mechanisms and Their Impact

Differential privacy is a cornerstone of privacy preservation in FL, with methods like DP-FedAvg adding noise to model updates to ensure privacy guarantees. However, DP introduces trade-offs: the added noise can reduce model accuracy and complicate the detection of malicious updates, impacting robustness. The survey "Balancing privacy and performance in federated learning: A systematic literature review on methods and metrics" (2024) synthesizes research on this issue, noting that privacy-preserving mechanisms increase computational and communication overheads, potentially compromising data utility and learning performance. Alternative approaches, such as secure multi-party computation and homomorphic encryption, are explored but also face computational complexity challenges.

### Robustness Against Adversarial Attacks

Robustness in FL focuses on mitigating the impact of malicious clients, particularly through Byzantine-robust aggregation techniques like Multi-Krum, Trimmed Mean, and Median. These methods filter out anomalous updates to ensure model integrity but often assume honest clients and do not account for privacy constraints. The user’s PRB-guided aggregation method addresses this gap by adaptively balancing privacy and robustness, using client trust scoring and gradient clipping. Experiments with datasets like CIFAR-10 and FEMNIST show that PRB-guided aggregation outperforms baselines like FedAvg and Multi-Krum in high-heterogeneity settings (Dirichlet α≤0.1), achieving 5-15% better trade-offs in privacy, robustness, and utility.

### Practical Methods to Balance Privacy and Robustness

Practical solutions aim to integrate privacy and robustness mechanisms. "A Privacy Robust Aggregation Method Based on Federated Learning in the IoT" (2023) proposes a Privacy Robust Aggregation (PBA) method that combines DP with denoising (using Kolmogorov–Smirnov distance) and outlier filtering (based on approximate Euclidean distance). PBA resists Deep Leakage from Gradients (DLG) attacks for privacy and maintains accuracy (85–88%) against Gaussian and Label Flipping attacks with up to 10% malicious clients. However, its performance degrades with higher malicious client fractions (30–60%), highlighting the limits of current methods under strong attacks.

### Comprehensive Surveys and Future Directions

Surveys provide a broad perspective on the privacy-robustness trade-off. "Privacy and Robustness in Federated Learning: Attacks and Defenses" (2020) (arXiv) offers a comprehensive taxonomy of threat models, poisoning attacks and defenses (robustness), and inference attacks and defenses (privacy). It emphasizes the inherent conflict between privacy and robustness, noting that robust defenses often require access to training data, which conflicts with privacy requirements. Similarly, "A Survey of Trustworthy Federated Learning with Perspectives on Security, Robustness, and Privacy" (2023) (arXiv) proposes a roadmap for trustworthy FL, summarizing efforts in security, robustness, and privacy and suggesting research directions like context-aware privacy budgets.

"Achieving security and privacy in federated learning systems: Survey, research challenges and future directions" (2021) analyzes the difficulty of reconciling security (including robustness) and privacy, proposing strategies like adaptive aggregation. "Trustworthy Federated Learning: Privacy, Security, and Beyond" (2024) (arXiv) extends this by surveying security and privacy issues, discussing defensive strategies, and identifying challenges in achieving robust security and privacy in FL. These surveys collectively highlight the need for adaptive, scalable solutions to balance privacy and robustness in real-world FL applications.

## Analysis of the Literature

### Strengths

- **Theoretical Rigor**: Frameworks like PRB and studies on LDP provide mathematically grounded insights into the trade-off, enabling precise optimization of privacy and robustness parameters.
- **Practical Innovations**: Methods like PBA and PRB-guided aggregation demonstrate feasible ways to integrate privacy and robustness, validated through experiments on real-world datasets.
- **Comprehensive Surveys**: Reviews from 2020 to 2024 offer structured syntheses of the field, identifying gaps and guiding future research.

### Limitations

- **Context-Specific Results**: Findings vary by dataset, model, and attack type, limiting generalizability. For instance, PBA’s effectiveness drops with high malicious client fractions.
- **Computational Overhead**: Privacy and robustness mechanisms often increase computational and communication costs, posing challenges for resource-constrained devices.
- **Limited Joint Optimisation**: Many studies focus on either privacy or robustness, with fewer addressing their joint optimization comprehensively.

### Gaps and Opportunities

- **Adaptive Mechanisms**: Developing adaptive privacy budgets that adjust based on attack strength or client heterogeneity could improve the trade-off.
- **Scalability**: More research is needed on how privacy-robustness methods scale with large client pools or extreme data heterogeneity, as explored in the user’s scalability analysis.
- **Real-World Applications**: Validating these methods in domains like healthcare requires further investigation.

## Future Directions

The literature and user-provided context suggest several directions for advancing research on the privacy-robustness trade-off in FL:

1. **Context-Aware Frameworks**: Developing frameworks that dynamically adjust privacy and robustness parameters based on real-time threat assessments.
2. **Hybrid Aggregation Methods**: Combining DP with advanced robust aggregation techniques, such as clustering-based methods, to enhance both objectives.
3. **Scalability and Heterogeneity**: Investigating how trade-offs hold in large-scale, highly heterogeneous FL systems.
4. **Interdisciplinary Approaches**: Integrating insights from cryptography, distributed systems, and ethics to design trustworthy FL systems compliant with regulations like GDPR.

## Conclusion

The privacy-robustness trade-off in federated learning is a complex challenge with significant implications for deploying FL in privacy-sensitive and attack-prone environments. Theoretical frameworks like PRB provide rigorous insights, while practical methods like PBA demonstrate feasible solutions. Surveys highlight the broader landscape, identifying challenges and opportunities. The expanded literature matrix below includes additional papers to provide a comprehensive view of the field, addressing the user’s request to increase the number of literatures while reducing the document’s writing.

## Matrix of Relevant Literature

| **Paper Title** | **Year** | **Type** | **Main Focus** | **Key Contributions on Privacy-Robustness Trade-off** | **Methods Used** | **Key Findings** |
| --- | --- | --- | --- | --- | --- | --- |
| Balancing privacy and performance in federated learning: A systematic literature review on methods and metrics | 2024 | Survey | Balancing privacy and performance in FL | Comprehensive overview of methods and metrics for privacy in FL, discussion on trade-offs between privacy and performance | Systematic review | Highlights impact of privacy mechanisms on FL performance, discusses challenges and open issues |
| Understanding the Interplay between Privacy and Robustness in Federated Learning | 2021 | Research Paper | Effect of LDP on robustness in FL | Theoretical and empirical analysis of how LDP affects adversarial robustness | Theoretical analysis, empirical experiments | LDP has both positive and negative effects on robustness |
| A Privacy Robust Aggregation Method Based on Federated Learning in the IoT | 2023 | Research Paper | Proposed PBA method for privacy and robustness | Integrates DP with denoising and outlier filtering | DP, denoising with KS distance, gradient encoding, outlier detection | Achieves privacy and robustness against up to 10% malicious clients |
| Achieving security and privacy in federated learning systems: Survey, research challenges and future directions | 2021 | Survey | Security and privacy in FL | Analyzes trade-off between security and privacy, discusses challenges and solutions | Survey, empirical evaluation | Discusses difficulties in achieving both security and privacy, provides ways to balance both |
| A Survey of Trustworthy Federated Learning with Perspectives on Security, Robustness, and Privacy | 2023 | Survey | Trustworthy FL with focus on security, robustness, privacy | Proposes roadmap for trustworthy FL, summarizes efforts on security, robustness, privacy | Survey | Discusses interrelations between security, robustness, and privacy, points out research directions |
| Privacy and Robustness in Federated Learning: Attacks and Defenses | 2020 | Survey | Privacy and robustness in FL, covering attacks and defenses | Comprehensively surveys threat models, poisoning attacks and defenses, inference attacks and defenses | Reviews various methods including DP, robust aggregation | Highlights the importance of balancing privacy and robustness in FL design |
| Federated Learning with Local Differential Privacy: Trade-offs between Privacy, Utility, and Communication | 2021 | Research Paper | Trade-offs in FL with LDP between privacy, utility, and communication | Theoretical analysis of trade-offs using appropriate metrics | Gaussian mechanisms for LDP, theoretical bounds | Stronger privacy leads to decreased utility and increased communication rate |
| Trustworthy Federated Learning: Privacy, Security, and Beyond | 2024 | Survey | Security and privacy in FL, with a focus on trustworthiness | Surveys security and privacy issues, defensive strategies, applications, and research directions | Reviews defensive strategies for security and privacy | Identifies security challenges in FL and proposes directions for secure and efficient FL systems |
