# MLprojectZINE

## DDoS Attack

DDoS, or Distributed Denial of Service, is a type of cyber attack where multiple systems are used to flood a target system with requests, overwhelming its ability to respond to legitimate requests. This can result in the target system becoming unavailable to its users.
In a DDoS attack, the attacker uses a botnet, which is a network of compromised systems. These systems are often compromised through malware or vulnerabilities in software. The attacker then sends a flood of requests to the target system, overwhelming its ability to respond.

## Strategies to prevent DDoS Attack

To mitigate the impact of DDoS attacks, organizations can implement various strategies, such as:

1) Using Content Delivery Networks (CDNs): CDNs can help distribute the traffic across multiple servers, reducing the load on the target system.

2) Implementing Traffic Shaping: Traffic shaping involves prioritizing certain types of traffic over others. This can help ensure that legitimate requests are not delayed due to the flood of malicious requests.

3) Using Web Application Firewalls (WAFs): WAFs can help filter out malicious requests before they reach the target system.

4) Regularly Updating Software: Ensuring that all software on the target system is up-to-date can help prevent vulnerabilities from being exploited in DDoS attacks.

5) Implementing Failover Mechanisms: Failover mechanisms can help ensure that the target system remains available even if it is overwhelmed by a DDoS attack.

## Project Details

In this project I have implemented five different models for DDoS detection. These algorithms are:

1) Support Vector Machine(SVM)
   
2) K-Nearest Neighbors(KNN)

3) Gaussian Naive Bayes(GNB)

4) Random Forests

5) Decision Trees

In this project dataset contains labelled data in which 'Normal' shows No-DDoS while other shows DDoS.

### Evaluation Metrics

**Accuracy**: This metric represents the fraction of correctly classified samples and is a common way to evaluate a model's performance.
  
  `Accuracy = (TP + TN) / (TP + TN + FP + FN)`

**Recall**: Also known as sensitivity, recall measures the true positive rate (TPR), indicating the proportion of actual positive values correctly identified.
  
  `Recall = TP / (TP + FN)`

**Precision**: Precision, or positive predictive value, measures the consistency of repeated measurements under unchanged conditions.
  
  `Precision = TP / (TP + FP)`

**F1 Score**: The F1 score, a harmonic average of recall and precision, is valuable when recall and precision conflict.
  
  `F1 Score = 2 * (Recall * Precision) / (Recall + Precision)`

**Confusion Matrix**: The confusion matrix compares predicted and actual values, providing insight into algorithm performance.

   
