# TML Assignment 1  
## Membership Inference Attack (MIA)

This assignment implements a **Membership Inference Attack (MIA)**. The goal is to infer whether a data point from a private dataset was part of the training set used to train the provided ResNet18 model. Our approach estimates the likelihood of a sample being a member by leveraging the softmax confidence distributions observed on a public dataset.

The implemented solution can be found in the [**mia.py**](mia.py) file. The submission file containing the membership scores (`test.csv`) can also be accessed from this repository.

---

## Data accessible to the adversary

- A **ResNet18 model** trained on an undisclosed dataset, with normalized data using:
  - Mean: [0.2980, 0.2962, 0.2987]
  - Std: [0.2886, 0.2875, 0.2889]
- A **public dataset** containing:
  - IDs, images, class labels, and **membership information** (1 == member, 0 == non-member)
- A **private dataset** containing:
  - IDs, images, class labels, and **no membership information** (membership = None)
- **No access** to the training data of the ResNet18 model
- **No access** to the true membership of the private dataset

---

## Approach used

The implemented approach is detailed in the [**mia.py**](LiRA_mia.py) file.  
We adopt the **Likelihood Ratio Attack (LiRA)** framework, which involves the following steps:

1. **Softmax confidence collection**:
   - For each sample in the public dataset, we compute its softmax confidence (probability assigned to the correct label).
   - We separate the confidence scores into **in** (members) and **out** (non-members) groups based on the public dataset’s membership labels.

    ```python
    conf_in = []
    conf_out = []
    with torch.no_grad():
        for i in range(len(public_data)):
            id_, img, label, member = public_data[i]
            img = transform(img).unsqueeze(0)
            output = model(img)
            prob = torch.softmax(output, dim=1)[0, label].item()
            if member == 1:
                conf_in.append(prob)
            else:
                conf_out.append(prob)
    ```

2. **Gaussian distribution modeling**:
   - We model the distribution of confidence scores for both **in** and **out** samples as Gaussian distributions:
     - \( p_{in} \sim \mathcal{N}(\mu_{in}, \sigma^2_{in}) \)
     - \( p_{out} \sim \mathcal{N}(\mu_{out}, \sigma^2_{out}) \)
    ```python
    mu_in, std_in = np.mean(conf_in), np.std(conf_in) + 1e-8
    mu_out, std_out = np.mean(conf_out), np.std(conf_out) + 1e-8
    ```

3. **Likelihood ratio computation**:
   - For each sample in the private dataset, we compute its softmax confidence and evaluate the likelihood ratio:
     \[
     \Lambda = \frac{p(\text{conf} | \mathcal{N}(\mu_{in}, \sigma^2_{in}))}{p(\text{conf} | \mathcal{N}(\mu_{out}, \sigma^2_{out}))}
     \]
   - This likelihood ratio is used as the **membership confidence score**.
   ```python
    scores = []
    with torch.no_grad():
        for i in range(len(private_data)):
            id_, img, label, _ = private_data[i]
            img = transform(img).unsqueeze(0)
            output = model(img)
            prob = torch.softmax(output, dim=1)[0, label].item()
            p_in = norm.pdf(prob, mu_in, std_in)
            p_out = norm.pdf(prob, mu_out, std_out)
            lira_score = p_in / (p_out + 1e-8)
            scores.append((id_, lira_score))
    ```


## Why we use this approach

- The LiRA attack leverages **proxy distributions** from the public dataset to estimate membership probabilities for the private dataset.
- It is **effective for scenarios where shadow model training is computationally expensive or unavailable**. 
- The attack is **model-independent**, relying solely on the outputs of the target model (softmax scores).

---

## Results


Using the approach, we submitted the `test.csv` file containing membership scores for the private dataset samples.  
Our results on the scoreboard are as follows:

| Metric                | Score                       |
|-----------------------|-----------------------------|
| TPR@FPR=0.05          | 0.06567                     |
| AUC                   | 0.64027                     |

---

## Other ideas on implementation that our approach has leveraged

- **Gaussian modeling** of confidence distributions.
- **Likelihood ratio** as a scoring function.
- **Softmax probabilities** as the primary feature for MIA.
- **Public dataset** as a proxy for distribution estimation.

---

## Files and their descriptions

| File           | Description                                             |
|----------------|---------------------------------------------------------|
| LiRA_mia.py    | Main implementation of the LiRA-based Membership Inference Attack |
| 01_MIA.pt      | Provided ResNet18 model checkpoint                       |
| pub_out.pt     | Public dataset with membership labels                   |
| priv_out.pt    | Private dataset without membership labels               |
| test.csv       | Submission file containing membership confidence scores |
| README.md      | Description of the approach and submission instructions  |
---