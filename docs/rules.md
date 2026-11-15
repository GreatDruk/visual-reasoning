By participating, you agree to these rules, the course academic-integrity policy, the current [Rating Statement](https://avkornaev.github.io/Intro-to-ML-2026/syllabus.html), and Kaggle’s Terms of Service.

This is an educational competition. There is no monetary prize.

## 1\. Schedule

All common deadlines are at **14:00 Moscow time (MSK, UTC+3)**. Assigned laboratory classes follow their timetables.

| Event | Date |
| --- | --- |
| Announcement | Wednesday, November 11, 2026 |
| Full competition materials and Kaggle page available; submissions disabled | Thursday, November 12, 2026 |
| Individual checkpoint work and upload of results | During the assigned laboratory class, November 12-14, 2026 |
| Kaggle submissions and team phase open | Saturday, November 14, 2026 |
| Team formation and Kaggle merger deadline | Wednesday, November 18, 2026 |
| Individual checkpoint defense | During the assigned laboratory class, November 19-21, 2026 |
| Final Kaggle submission and Moodle package deadline | Saturday, November 22, 2026 |
| Verified results from the private part published | Wednesday, November 25, 2026 |
| Optional team presentations | During the assigned laboratory class, November 26-28 2026 |

The Kaggle deadline controls leaderboard submissions. The Moodle deadline controls the course deliverable. Both are required.

## 2\. Individual checkpoint

Each student starts the individual checkpoint during their assigned laboratory class on November 12-14, uploads the notebook and results by the end of that class, and defends the work during their assigned class on November 19-21.

For the individual checkpoint, every student must:

-   attend their assigned laboratory class;
-   independently finish a solution model by completing lab assignments providing an AUC-ROC score as the result;
-   submit the required notebook and results through their own Moodle account;
-   be able to explain and reproduce the solution.

General discussion is permitted, but students must not exchange code or prediction files during the individual phase.

Submitting the individual work is required before joining a team or entering the team phase. Checkpoint credit and leaderboard-bonus eligibility also require a successful individual defense the following week.

## 3\. Teams

After submitting their individual checkpoint, students may continue alone or form teams of up to **three** students.

-   Every member must be enrolled in the course, have submitted the individual checkpoint, and complete its defense for eligibility.
-   Teams must be finalized in Kaggle by the merger deadline.
-   The final Moodle team list must exactly match the Kaggle team membership.
-   Team membership is final after the merger deadline.
-   Collaboration is permitted only within the declared team.
-   Code, engineered features, predictions and private competition materials must not be exchanged between teams.
-   Competition data and invitation links must not be shared outside the course or posted publicly.
-   Every member must make a meaningful contribution and be able to explain the complete solution.
-   Guest, ghost, gifted or honorary membership is prohibited.

The organizers may assess individual understanding separately. A student who cannot explain the submitted work may lose individual credit even if the team submission remains valid.

## 4\. Allowed methods

Participants may use standard CPU or GPU-compatible machine learning methods for Deep Learning. You can go through it using Pytorch or TensorFlow. Our Lab solutions will be based on Pytorch methods.

In order to design a model you are allowed to use two kind of architectures that transform a specific modality into a vector. In practice, it insists to use two type of models: image-based and text-based architectures, then combine their results as a source of information about described entities.

Models that are ALLOWED to use:

1.  Image encoders: classification models from the torchvision list ([https://docs.pytorch.org/vision/0.22/models.html](https://docs.pytorch.org/vision/0.22/models.html)) with their published ImageNet weights, frozen or fine-tuned. You may use any part of them (e.g. intermediate feature maps).
    -   **BUT:** pretrained detection, segmentation, video and optical-flow models from that list (e.g. Faster R-CNN, Mask R-CNN, DeepLabV3) ARE NOT ALLOWED: their weights were trained to find or outline objects on other datasets.
2.  Text encoders: any text-only model that returns vector representations: RNN / LSTM / GRU, or pretrained encoder-only Transformers (BERT, etc.), run locally.
3.  CLIP / SigLIP / OpenCLIP dual encoders are allowed (zero-shot, fine-tuned or as feature extractors): each modality is encoded separately.
4.  Any layers you design and train yourself on the contest data ARE ALLOWED: fusion (e.g. FiLM), attention, layers that use positions in the feature map, and the final layers that predict whether a caption is true.

NOT ALLOWED methods and models:

1.  Pretrained models that process images and text together or generate one from the other: LLaVA, Qwen-VL, GPT-4o, Gemini, BLIP / BLIP-2, ALBEF, ViLT, FLAVA, OWL-ViT, Grounding DINO, captioning and VQA models, or similar.
2.  LLMs or other text generators, including for creating extra captions.
3.  External data of similar type (images, scene files, questions) as well as models trained on similar datasets.
4.  Any methods that are trying to match contest images to external images.
5.  Manual labelling of test images, and any information about hidden labels.
6.  Pre-fitted models or predictions produced outside the declared team.

**BUT:** with the given data you are ALLOWED to do the following:

-   training on train.csv AND val.csv (e.g. a final model trained on both);
-   ensembles: combining predictions of several of your own models (e.g. averaging, or different models for different categories);
-   data augmentation, and extra captions created from the given training data;
-   using the unlabelled test images and captions (e.g. finding twin captions).

As for the augmentation part, you should be careful, as NOT every method can be used straightly with the dataset. **Why?**

| Augmentation type | Safe? | Reason |
| --- | --- | --- |
| Horizontal flip | Only if the caption is changed too | “left” and “right” must be swapped in the caption, otherwise the label becomes wrong |
| Vertical flip, rotation | No | Breaks “in front of” / “behind” and makes scenes physically impossible |
| Random crop, zoom-in | Risky | An object at the edge can be cut off, so counts and “there is…” captions may become false |
| Colour / hue jitter | No | “red” can turn into “purple”: attribute, binding and comparison labels break |
| Small brightness / contrast change, light blur, noise | Yes | Objects and colours stay recognisable |
| Text: reorder clauses (“A to the left of B” → “to the left of B there is A”) | Yes | Same meaning; helps with the test-only phrasings |
| New false captions by logic (n → n±1, left → right) | Yes | Labels are certain, so it’s free extra data |

Don't forget to fix your random seeds to make results reproducible.

For the **individual checkpoint**, design and fit proposed models on `train.csv` only; use `validation.csv` to evaluate models from Assignments 1 and 2. During the team phase, students may use the supplied labeled training and validation data for model development, including refitting on their combination. Report validation ROC-AUC only for experiments evaluated on a validation set that was held out from fitting; describe how the final model was trained.


## 5\. AI assistants

AI coding assistants and agents are allowed. However:

-   students remain responsible for every line of submitted code;
-   every computational step used to produce the submission must appear in the submitted package;
-   students must understand, reproduce and explain the solution;
-   AI assistance does not permit external data, models, segmentations or any other action with a dataset.

Chat histories are not required. Generated code is treated as the student’s own submitted work for academic-integrity purposes.

## 6\. Intended solution and prohibited conduct

The intended task is to learn a relationship between two modalities (text and images) from the supplied labeled examples.

Any method whose primary purpose is to recover concealed information, exploit an unintended property of the dataset or platform, or obtain a score without solving the intended prediction problem is PROHIBITED, even if the method is not explicitly listed here.

The following are also prohibited:

-   obtaining or reconstructing hidden targets or concealed source information;
-   manually editing test predictions using external information;
-   exploiting Kaggle, Moodle, scoring, file-format or API vulnerabilities;
-   using multiple Kaggle accounts or circumventing submission limits;
-   exchanging code, engineered features, labels or predictions between teams;
-   making competition data or invitation links publicly available.

Ordinary feature engineering and diagnostic analysis of the supplied labeled data ARE ALLOWED within the roles defined in Section 4. If it is unclear whether a method solves the intended problem, ask the instructors before using it.

## 7\. Kaggle submissions

-   Submissions are evaluated using ROC-AUC score.
-   The public leaderboard provides provisional feedback only.
-   Final ranking uses the private leaderboard after eligibility and reproducibility checks.
-   Each team may make at most **five submissions per day**.
-   Each team must designate **one final submission**.
-   Every submission must contain each test sample id exactly once and a numerical `target` probability value between 0 and 1.

The final submission must be generated by the team’s submitted code. Manual alteration of the prediction file **is prohibited**.

The team must record the designated Kaggle submission in its Moodle package. If Kaggle permits the selection of more than one final submission, only the submission identified in Moodle will be used for course assessment and bonuses.

## 8\. Moodle deliverables and reproducibility

Each student submits the individual checkpoint through Moodle.

For the team phase, one designated member submits one final Moodle package on behalf of the team. The package must identify every team member and contain:

-   the Kaggle team name;
-   the team-member list and a short contribution statement;
-   the designated Kaggle submission identifier or filename;
-   one executable notebook;
-   any necessary source files;
-   the corresponding `final_submission.csv`;
-   a short experiment table containing the model architecture, validation Accuracy, runtime and seed;
-   package and library versions.

The notebook must:

-   run from a fresh kernel using **Restart and Run All**;
-   train the final model and regenerate `final_submission.csv`;
-   use relative paths and the supplied competition files;
-   set seeds for Python, NumPy and every model;
-   contain all preprocessing, feature selection/engineering, training, ensembling, calibration and prediction steps;
-   run without internet access or hidden local files;
-   complete training and prediction within **50 minutes on Kaggle GPU (Kaggle Default Hardware)**;
-   use no more VRAM than it is allowed by **Kaggle Default Hardware** for a final solution.

The submitted notebook and source files must not exceed **20 MB**, excluding the supplied competition data and generated prediction CSV. Any external datasets or image additions excluding augmentations are **prohibited**.

The regenerated `id` values of images must match exactly as well labels from `classes.csv`.

## 9\. Assessment and bonuses

Within the non-leaderboard assessment for this competition:

-   **40%** — individual checkpoint and demonstrated understanding;
-   **35%** — correctness and reproducibility of the final pipeline;
-   **25%** — experiment quality, required model comparison and justification of the final choice.

Leaderboard bonuses are based on the verified private-leaderboard ranking among eligible teams. Bonus tiers, point values, eligibility requirements, presentation bonuses and applicable limits are defined in the current [Rating Statement](https://avkornaev.github.io/Intro-to-ML-2026/syllabus.html).

## 10\. Verification and enforcement

The organizers may:

-   rerun submitted notebooks;
-   compare regenerated and submitted predictions;
-   inspect submission history and team membership;
-   request a short individual explanation or demonstration;
-   exclude an unreproducible or rule-breaking submission from leaderboard bonuses;
-   refer suspected academic misconduct through the normal course procedure.

Clarifications may be requested after the deadline, but code, models and predictions may not be replaced.

Personal hardware, network, account or time-management problems do not normally justify a late submission. If an official Kaggle or Moodle outage materially affects participants, the organizers may extend the deadline for everyone.

## ENTRY IN THIS COMPETITION CONSTITUTES YOUR ACCEPTANCE OF THESE OFFICIAL COMPETITION RULES.

### A. LEGALLY BINDING AGREEMENT

The Competition named below is a skills-based competition to promote and further the field of data science. You must register via the Competition website (“**Competition Website**”) to enter. To enter the Competition, you must agree to these Official Competition Rules, which incorporate by reference the provisions and content of the Competition Website and any Specific Competition Rules herein (collectively, the "**Rules**"). Please read these Rules carefully before entry to ensure you understand and agree. You further agree that Submission (as defined below) in the Competition constitutes agreement to these Rules. You may not submit to the Competition and are not eligible to receive the prizes associated with this Competition unless you agree to these Rules. These Rules form a binding legal agreement between you and the Competition Sponsor with respect to the Competition. Your competition Submissions must conform to the requirements stated on the Competition Website. Your Submissions will be scored based on the evaluation metric described on the Competition Website. Subject to compliance with the Competition Rules, Prizes, if any, will be awarded to Participants (as defined below) with the best scores, based on the merits of the data science models submitted.

**You cannot sign up to Kaggle.com from multiple accounts and therefore you cannot enter or submit from multiple accounts.**

Unless otherwise stated in the Competition-Specific Rules below or prohibited by internal policies of the Competition Entities, employees, interns, contractors, officers and directors of Competition Entities may enter and participate in the Competition, but are not eligible to win any Prizes. "**Competition Entities**" means the Competition Sponsor, Kaggle Inc. (“**Kaggle**”), and their respective parent companies, subsidiaries and affiliates. If you are such a Participant from a Competition Entity, you are subject to all applicable internal policies of your employer with respect to your participation.

### B. COMPETITION-SPECIFIC TERMS AND RULES

**COMPETITION TITLE:** \[INSERT\]

**COMPETITION SPONSOR:** \[INSERT\]

**COMPETITION SPONSOR ADDRESS:** \[INSERT\]

**COMPETITION WEBSITE:** [https://www.kaggle.com/competitions/\[INSERT](https://www.kaggle.com/competitions/[INSERT)\]

**TOTAL PRIZES AVAILABLE:** $\[INSERT\]

**First Prize:** $\[INSERT\]

**Second Prize:** $\[INSERT\]

**Third Prize:** $\[INSERT\]

\[INSERT NON-MONETARY PRIZES AS APPLICABLE\]

**WINNER LICENSE TYPE:** \[INSERT\]

**DATA ACCESS AND USE:** \[INSERT\]

In addition to the provisions of the General Competition Rules below, you understand and agree to the Competition-Specific Rules required by the Competition Sponsor as set forth below:

#### 1\. TEAM LIMITS

a. The maximum Team (as defined below) size is five (5).

b. Team mergers are allowed and can be performed by the Team leader. In order to merge, the combined Team must have a total Submission count less than or equal to the maximum allowed as of the Team Merger Deadline (as described below). The maximum allowed is the number of Submissions per day multiplied by the number of days the competition has been running.

#### 2\. SUBMISSION LIMITS

a. You may submit a maximum of five (5) Submissions per day.

b. You may select up to two (2) Final Submissions (as defined below) for judging.

#### 3\. COMPETITION TIMELINE

Competition Timeline dates (including Entry Deadline, Final Submission Deadline, Start Date, and Team Merger Deadline, as applicable) are set forth on the Competition’s Overview > Timeline page.

#### 4\. COMPETITION DATA

a. Data Access and Use. \[CHOOSE ONE OF THE 3 OPTIONS BELOW BASED ON WHAT THE SPONSOR CHOOSES IN THE SOW.\]

\[_Competition Use Only is checked_: You may access and use the Competition Data only for participating in the Competition and on Kaggle.com forums. The Competition Sponsor reserves the right to disqualify any Participant who uses the Competition Data other than as permitted by the Competition Website and these Rules.\]

\[_Competition Use and Non-Commercial & Academic Research is checked_: You may access and use the Competition Data for non-commercial purposes only, including participation in the Competition and on Kaggle.com forums, and for academic research and education. The Competition Sponsor reserves the right to disqualify any Participant who uses the Competition Data other than as permitted by the Competition Website and these Rules.\]

\[_Competition Use and Commercial Purposes are checked_: You may access and use the Competition Data for any purpose, whether commercial or non-commercial, including for participating in the Competition and on Kaggle.com forums, and for academic research and education. The Competition Sponsor reserves the right to disqualify any Participant who uses the Competition Data other than as permitted by the Competition Website and these Rules.\]

\[_ADD IN THE FOLLOWING, IN ADDITION TO THE CHOSEN SECTION ABOVE, IF SPECIFIC LICENSE IS CHECKED_: The Competition Data is also subject to the following terms and conditions: \[INSERT URL\], OR

\[None. Competition Data will not be provided by Competition Sponsor for this Competition.\]

b. **Data Security**. You agree to use reasonable and suitable measures to prevent persons who have not formally agreed to these Rules from gaining access to the Competition Data. You agree not to transmit, duplicate, publish, redistribute or otherwise provide or make available the Competition Data to any party not participating in the Competition. You agree to notify Kaggle immediately upon learning of any possible unauthorized transmission of or unauthorized access to the Competition Data and agree to work with Kaggle to rectify any unauthorized transmission or access.

#### 5\. WINNERS’ LICENSE \[INSERT ONLY THE RELEVANT LICENSE LANGUAGE FOR THE COMPETITION.\]

a. In furtherance of Section C(6) below, you hereby grant and will grant the Competition Sponsor the following license(s) with respect to your Submission if you are a Competition winner:

\[_Non-Exclusive_: You hereby grant and will grant to Competition Sponsor and its designees a worldwide, non-exclusive, sub-licensable, transferable, fully paid-up, royalty-free, perpetual, irrevocable right to use, reproduce, distribute, create derivative works of, publicly perform, publicly display, digitally perform, make, have made, sell, offer for sale and import your winning Submission and the source code used to generate the Submission, in any media now known or developed in the future, for any purpose whatsoever, commercial or otherwise, without further approval by or payment to you.\]

\[_Open Source_: You hereby license and will license your winning Submission and the source code used to generate the Submission under Open Source License CC BY 4.0 or Apache 2.0. In the event that input data or pretrained models with an incompatible license are used to generate your winning solution, you do not need to grant an open source license in the preceding sentence for that data and/or model(s).\]

b. For generally commercially available software that you used to generate your Submission that is not owned by you, but that can be procured by the Competition Sponsor at a cost that does not exceed the amount of a prize in the competition, you do not need to grant the license in the preceding Section for that software.

c. You may be required by the Sponsor to provide a detailed description of how the winning Submission was generated, to the Competition Sponsor’s specifications, as outlined in Section B(7) (_Winners’ Obligations_). This may include a detailed description of methodology, where one must be able to reproduce the approach by reading the description, and includes a detailed explanation of the architecture, preprocessing, loss function, training details, hyper-parameters, etc. The description should also include a link to a code repository with complete and detailed instructions so that the results obtained can be reproduced.

#### 6\. EXTERNAL DATA AND TOOLS

a. Subject to Subsection (b) below, you may use data other than the Competition Data (“**External Data**”) to develop and test your Submissions. However, you will ensure the External Data is either publicly available and equally accessible to use by all Participants of the Competition for purposes of the competition at no cost to the other Participants, or satisfies the Reasonableness Standard as described in Subsection (b) below. The ability to use External Data under this Section does not limit your other obligations under these Competition Rules, including but not limited to Section B(7) (_Winners’ Obligations_).

b. The use of External Data and models is acceptable unless specifically prohibited by the Competition Host. Because of the potential costs or restrictions (e.g., territorial restrictions or “geo restrictions”) associated with obtaining rights to use External Data or certain software and associated tools, their use must meet the Reasonableness Standard as described below. Also, regardless of the cost challenges as they might affect all Participants during the course of the competition, the costs of potentially procuring a license for software used to generate a Submission, must also be considered. The Competition Host will employ an assessment of whether or not the following criteria can exclude the use of the particular LLM, dataset(s), or tool(s):

i. Are Participants being excluded from a competition because of the "excessive" costs for access to certain LLMs, external data, or tools that might be used by other Participants. The Competition Host will assess the excessive cost concern by applying a “reasonableness” standard (the “**Reasonableness Standard**”). The Reasonableness Standard will be determined by the Competition Host in consultation with Kaggle and applied by the Competition Host in light of certain criteria, including but not limited to, cost thresholds and accessibility. By way of example only, purchasing a license to use a proprietary dataset that exceeds the cost of a prize in the competition may not be considered reasonable.

c. Automated Machine Learning Tools

i. Individual Participants and Teams may use automated machine learning tool(s) (“**AMLT**”) (e.g., Google AutoML, H2O Driverless AI, etc.) to create a Submission, provided that the Participant or Team ensures that they have an appropriate license to the AMLT such that they are able to comply with the Competition Rules.

#### 7\. WINNERS’ OBLIGATIONS

a. As a condition to being awarded a Prize, a Prize winner must fulfill the following obligations:

i. Deliver to the Competition Sponsor the final model's software code as used to generate the winning Submission and associated documentation. The delivered software code should follow [these documentation guidelines](https://www.kaggle.com/WinningModelDocumentationGuidelines), must be capable of generating the winning Submission, and contain a description of resources required to build and/or run the executable code successfully. For avoidance of doubt, delivered software code should include training code, inference code, and a description of the required computational environment.

ii. To the extent that the final model’s software code includes generally commercially available software that is not owned by you, but that can be procured by the Competition Sponsor at a cost that does not exceed the amount of a prize in the competition, then instead of delivering the code for that software to the Competition Sponsor, you must identify that software, method for procuring it, and any parameters or other information necessary to replicate the winning Submission; Individual Participants and Teams who create a Submission using an AMLT may win a Prize. However, for clarity, the potential winner’s Submission must still meet the requirements of these Rules, including but not limited to Section B(5) (_Winners’ License_), Section B(7) (_Winners’ Obligations_), and Section C(14) (_Warranty, Indemnity, and Release_).

iii. Individual Participants and Teams who create a Submission using an AMLT may win a Prize. However, for clarity, the potential winner’s Submission must still meet the requirements of these Rules.

b. Grant to the Competition Sponsor the license to the winning Submission stated in the Competition Specific Rules above, and represent that you have the unrestricted right to grant that license;

c. Sign and return all Prize acceptance documents as may be required by Competition Sponsor or Kaggle, including without limitation: (a) eligibility certifications; (b) licenses, releases and other agreements required under the Rules; and (c) U.S. tax forms (such as IRS Form W-9 if U.S. resident, IRS Form W-8BEN if foreign resident, or future equivalents).

#### 8\. GOVERNING LAW

Unless otherwise provided in the Competition Specific Rules above, all claims arising out of or relating to these Rules will be governed by California law, excluding its conflict of laws rules, and will be litigated exclusively in the Federal or State courts of Santa Clara County, California, USA. The parties consent to personal jurisdiction in those courts. If any provision of these Rules is held to be invalid or unenforceable, all remaining provisions of the Rules will remain in full force and effect.

### C. GENERAL COMPETITION RULES

#### 1\. ELIGIBILITY

a. To be eligible to enter the Competition, you must be (i) a registered account holder at Kaggle.com; (ii) the older of 18 years old or the age of majority in your jurisdiction of residence (unless otherwise agreed to by Competition Sponsor and appropriate parental/guardian consents have been obtained by Competition Sponsor); (iii) not a resident of Crimea, so-called Donetsk People's Republic (DNR) or Luhansk People's Republic (LNR), Cuba, Iran, or North Korea; and (iv) not a person or representative of an entity under U.S. export controls or sanctions (see [https://ofac.treasury.gov/](https://ofac.treasury.gov/)).

b. Competitions are open to residents of the United States and worldwide, except that if you are a resident of Crimea, so-called Donetsk People's Republic (DNR) or Luhansk People's Republic (LNR), Cuba, Iran, North Korea, or are subject to U.S. export controls or sanctions, you may not enter the Competition. Other local rules and regulations may apply to you, so please check your local laws to ensure that you are eligible to participate in skills-based competitions. The Competition Host reserves the right to forego or award alternative Prizes where needed to comply with local laws. If a winner is located in a country where prizes cannot be awarded, then they are not eligible to receive a prize.

c. If you are entering as a representative of a company, educational institution or other legal entity, or on behalf of your employer, these rules are binding on you, individually, and the entity you represent or where you are an employee. If you are acting within the scope of your employment, or as an agent of another party, you warrant that such party or your employer has full knowledge of your actions and has consented thereto, including your potential receipt of a Prize. You further warrant that your actions do not violate your employer's or entity's policies and procedures.

d. The Competition Sponsor reserves the right to verify eligibility and to adjudicate on any dispute at any time. If you provide any false information relating to the Competition concerning your identity, residency, mailing address, telephone number, email address, ownership of right, or information required for entering the Competition, you may be immediately disqualified from the Competition.

#### 2\. SPONSOR AND HOSTING PLATFORM

The Competition is sponsored by Competition Sponsor named above. The Competition is hosted on behalf of Competition Sponsor by Kaggle. Kaggle is an independent contractor of Competition Sponsor, and is not a party to this or any agreement between you and Competition Sponsor. You understand that Kaggle has no responsibility with respect to selecting the potential Competition winner(s) or awarding any Prizes. Kaggle will perform certain administrative functions relating to hosting the Competition, and you agree to abide by the provisions relating to Kaggle under these Rules. As a Kaggle.com account holder and user of the Kaggle competition platform, remember you have accepted and are subject to the [Kaggle Terms of Use](http://www.kaggle.com/terms) in addition to these Rules.

#### 3\. COMPETITION PERIOD

For the purposes of Prizes, the Competition will run from the Start Date and time to the Final Submission Deadline (such duration, the “**Competition Period**”). The Competition Timeline is subject to change, and Competition Sponsor may introduce additional hurdle deadlines during the Competition Period. Any updated or additional deadlines will be publicized on the Competition Website. It is your responsibility to check the Competition Website regularly to stay informed of any deadline changes. YOU ARE RESPONSIBLE FOR DETERMINING THE CORRESPONDING TIME ZONE IN YOUR LOCATION.

#### 4\. COMPETITION ENTRY

a. NO PURCHASE REQUIRED TO ENTER OR WIN. To enter the Competition, you must register on the Competition Website prior to the Entry Deadline, and follow the instructions for developing and entering your Submission through the Competition Website. Your Submissions must be made in the manner and format, and in compliance with all other requirements, stated on the Competition Website (the "**Requirements**"). Submissions must be received before any Submission deadlines stated on the Competition Website. Submissions not received by the stated deadlines will not be eligible to receive a Prize.

b. Submissions may not use or incorporate information from hand labeling or human prediction of the validation dataset or test data records.

c. If the Competition is a multi-stage competition with temporally separate training and/or test data, one or more valid Submissions may be required during each Competition stage in the manner described on the Competition Website in order for the Submissions to be Prize eligible.

d. Submissions are void if they are in whole or part illegible, incomplete, damaged, altered, counterfeit, obtained through fraud, or late. Competition Sponsor reserves the right to disqualify any entrant who does not follow these Rules, including making a Submission that does not meet the Requirements.

#### 5\. INDIVIDUALS AND TEAMS

a. Individual Account. You may make Submissions only under one unique Kaggle.com account. You will be disqualified if you make Submissions through more than one Kaggle.com account, or attempt to falsify an account to act as your proxy. You may submit up to the maximum number of Submissions per day as specified on the Competition Website.

b. Teams. If permitted under the Competition Website guidelines, multiple individuals may collaborate as a Team; however, you may join or form only one Team. Each Team member must be a single individual with a separate Kaggle.com account. You must register individually for the Competition before joining a Team. You must confirm your Team membership to make it official by responding to the Team notification message sent to your Kaggle.com account. Team membership may not exceed the Maximum Team Size stated on the Competition Website.

c. Team Merger. Teams (or individual Participants) may request to merge via the Competition Website. Team mergers may be allowed provided that: (i) the combined Team does not exceed the Maximum Team Size; (ii) the number of Submissions made by the merging Teams does not exceed the number of Submissions permissible for one Team at the date of the merger request; (iii) the merger is completed before the earlier of: any merger deadline or the Competition deadline; and (iv) the proposed combined Team otherwise meets all the requirements of these Rules.

d. Private Sharing. No private sharing outside of Teams. Privately sharing code or data outside of Teams is not permitted. It's okay to share code if made available to all Participants on the forums.

#### 6\. SUBMISSION CODE REQUIREMENTS

a. Private Code Sharing. Unless otherwise specifically permitted under the Competition Website or Competition Specific Rules above, during the Competition Period, you are not allowed to privately share source or executable code developed in connection with or based upon the Competition Data or other source or executable code relevant to the Competition (“**Competition Code**”). This prohibition includes sharing Competition Code between separate Teams, unless a Team merger occurs. Any such sharing of Competition Code is a breach of these Competition Rules and may result in disqualification.

b. Open Source. You hereby license your Submission and the source code used to generate the Submission under the Open Source Initiative license presented on the Competition Website (see [http://www.opensource.org](http://www.opensource.org/)) that in no event limits commercial use of such code or model containing or depending on such code. In the event that input data or pretrained models with an incompatible license are used to generate your winning solution, you do not need to grant an open source license in the preceding sentence for that data and/or model(s).

c. For generally commercially available software that you used to generate your Submission that is not owned by you, but that can be procured by the Competition Sponsor at a cost that does not exceed the amount of a prize in the competition, you do not need to grant the license in the preceding Section for that software.

#### 7\. DETERMINING WINNERS

a. Each Submission will be scored and/or ranked by the evaluation metric, stated on the Competition Website. During the Competition Period, the current ranking will be visible on the Competition Website's Public Leaderboard. The potential winner(s) are determined solely by the leaderboard ranking on the Private Leaderboard, subject to compliance with these Rules. The Public Leaderboard will be based on the public test set and the Private Leaderboard will be based on the private test set.

b. In the event of a tie, the Submission that was entered first to the Competition will be the winner. In the event a potential winner is disqualified for any reason, the Submission that received the next highest score rank will be chosen as the potential winner.

#### 8\. NOTIFICATION OF WINNERS & DISQUALIFICATION

a. The potential winner(s) will be notified by email.

b. If a potential winner (i) does not respond to the notification attempt within one (1) week from the first notification attempt or (ii) notifies Kaggle within one week after the Final Submission Deadline that the potential winner does not want to be nominated as a winner or does not want to receive a Prize, then, in each case (i) and (ii) such potential winner will not receive any Prize, and an alternate potential winner will be selected from among all eligible entries received based on the Competition’s judging criteria.

c. In case (i) and (ii) above Kaggle may disqualify the Participant. However, in case (ii) above, if requested by Kaggle, such potential winner may provide code and documentation to verify the Participant’s compliance with these Rules. If the potential winner provides code and documentation to the satisfaction of Kaggle, the Participant will not be disqualified pursuant to this paragraph.

d. Competition Sponsor reserves the right to disqualify any Participant from the Competition if the Competition Sponsor reasonably believes that the Participant has attempted to undermine the legitimate operation of the Competition by cheating, deception, or other unfair playing practices or abuses, threatens or harasses any other Participants, Competition Sponsor or Kaggle.

e. A disqualified Participant may be removed from the Competition leaderboard, at Kaggle's sole discretion. If a Participant is removed from the Competition Leaderboard, additional winning features associated with the Kaggle competition platform, for example Kaggle points or medals, may also not be awarded.

f. The final leaderboard list will be publicly displayed at Kaggle.com. Determinations of Competition Sponsor are final and binding.

#### 9\. PRIZES

a. Prize(s) are as described on the Competition Website and are only available for winning during the time period described on the Competition Website. The odds of winning any Prize depends on the number of eligible Submissions received during the Competition Period and the skill of the Participants.

b. All Prizes are subject to Competition Sponsor's review and verification of the Participant’s eligibility and compliance with these Rules, and the compliance of the winning Submissions with the Submissions Requirements. In the event that the Submission demonstrates non-compliance with these Competition Rules, Competition Sponsor may at its discretion take either of the following actions: (i) disqualify the Submission(s); or (ii) require the potential winner to remediate within one week after notice all issues identified in the Submission(s) (including, without limitation, the resolution of license conflicts, the fulfillment of all obligations required by software licenses, and the removal of any software that violates the software restrictions).

c. A potential winner may decline to be nominated as a Competition Winner in accordance with Section C(8)(b).

d. Potential winners must return all required Prize acceptance documents within two (2) weeks following notification of such required documents, or such potential winner will be deemed to have forfeited the prize and another potential winner will be selected. Prize(s) will be awarded within approximately thirty (30) days after receipt by Competition Sponsor or Kaggle of the required Prize acceptance documents. Transfer or assignment of a Prize is not allowed.

e. You are not eligible to receive any Prize if you do not meet the Eligibility requirements in Section C(1).

f. If a Team wins a monetary Prize, the Prize money will be allocated in even shares between the eligible Team members, unless the Team unanimously opts for a different Prize split and notifies Kaggle before Prizes are issued.

#### 10\. TAXES

ALL TAXES IMPOSED ON PRIZES ARE THE SOLE RESPONSIBILITY OF THE WINNERS. Payments to potential winners are subject to the express requirement that they submit all documentation requested by Competition Sponsor or Kaggle for compliance with applicable state, federal, local and foreign (including provincial) tax reporting and withholding requirements. Prizes will be net of any taxes that Competition Sponsor is required by law to withhold. If a potential winner fails to provide any required documentation or comply with applicable laws, the Prize may be forfeited and Competition Sponsor may select an alternative potential winner. Any winners who are U.S. residents will receive an IRS Form-1099 in the amount of their Prize.

#### 11\. GENERAL CONDITIONS

You will comply with all applicable federal, state, provincial and local laws and regulations.

#### 12\. PUBLICITY

You grant to Competition Sponsor and Kaggle and its affiliates an unrestricted, sublicensable, assignable, irrevocable, perpetual, worldwide, royalty-free license to your name, voice, image, persona, likeness, and performance in any audio, visual, and audiovisual recordings (including, but not limited to, photographs, digital images, digital video, digital audio, video tape, and audio tape) taken or to be taken of you by or on behalf of Kaggle (the “**Content**”) in the Competition listed above. This license includes without limitation the right to (i) reproduce, create derivative works of, and otherwise use the Content, in whole or in part, in any manner or in combination with any other material (the “**Works**”); and (ii) publicly display, publicly perform, sell, rent, distribute (directly or indirectly), transmit, or broadcast the Works.You further acknowledge that this license grants Kaggle the right to include Content in end-of-the-year recaps, Kaggle events, award show submissions, and fixed media for award shows. You waive all rights and release Kaggle from, and hold Kaggle harmless from, any claim, whether now known or unknown, relating to the use and exploitation of the Content.

#### 13\. PRIVACY

a. You acknowledge and agree that Competition Sponsor and Kaggle may collect, store, share and otherwise use personally identifiable information provided by you during the Kaggle.com account registration process and the Competition, including but not limited to, name, mailing address, phone number, and email address (“**Personal Information**”). Kaggle acts as an independent controller with regard to its collection, storage, sharing, and other use of this Personal Information, and will use this Personal Information in accordance with its [Privacy Policy](http://www.kaggle.com/privacy), including for administering the Competition. As a Kaggle.com account holder, you have the right to request access to, review, rectification, portability or deletion of any personal data held by Kaggle about you by logging into your account and/or contacting [Kaggle Support](http://www.kaggle.com/contact).

b. As part of Competition Sponsor performing this contract between you and the Competition Sponsor, Kaggle will transfer your Personal Information to Competition Sponsor, which acts as an independent controller with regard to this Personal Information. As a controller of such Personal Information, Competition Sponsor agrees to comply with all U.S. and foreign data protection obligations with regard to your Personal Information. Kaggle will transfer your Personal Information to Competition Sponsor in the country specified in the Competition Sponsor Address listed above, which may be a country outside the country of your residence. Such country may not have privacy laws and regulations similar to those of the country of your residence.

#### 14\. WARRANTY, INDEMNITY AND RELEASE

a. You warrant that your Submission is your own original work and, as such, you are the sole and exclusive owner and rights holder of the Submission, and you have the right to make the Submission and grant all required licenses. You agree not to make any Submission that: (i) infringes any third party proprietary rights, intellectual property rights, industrial property rights, personal or moral rights or any other rights, including without limitation, copyright, trademark, patent, trade secret, privacy, publicity or confidentiality obligations, or defames any person; or (ii) otherwise violates any applicable U.S. or foreign state or federal law.

b. To the maximum extent permitted by law, you indemnify and agree to keep indemnified Competition Entities at all times from and against any liability, claims, demands, losses, damages, costs and expenses resulting from any of your acts, defaults or omissions and/or a breach of any warranty set forth herein. To the maximum extent permitted by law, you agree to defend, indemnify and hold harmless the Competition Entities from and against any and all claims, actions, suits or proceedings, as well as any and all losses, liabilities, damages, costs and expenses (including reasonable attorneys fees) arising out of or accruing from: (i) your Submission or other material uploaded or otherwise provided by you that infringes any third party proprietary rights, intellectual property rights, industrial property rights, personal or moral rights or any other rights, including without limitation, copyright, trademark, patent, trade secret, privacy, publicity or confidentiality obligations, or defames any person; (ii) any misrepresentation made by you in connection with the Competition; (iii) any non-compliance by you with these Rules or any applicable U.S. or foreign state or federal law; (iv) claims brought by persons or entities other than the parties to these Rules arising from or related to your involvement with the Competition; and (v) your acceptance, possession, misuse or use of any Prize, or your participation in the Competition and any Competition-related activity.

c. You hereby release Competition Entities from any liability associated with: (a) any malfunction or other problem with the Competition Website; (b) any error in the collection, processing, or retention of any Submission; or (c) any typographical or other error in the printing, offering or announcement of any Prize or winners.

#### 15\. INTERNET

Competition Entities are not responsible for any malfunction of the Competition Website or any late, lost, damaged, misdirected, incomplete, illegible, undeliverable, or destroyed Submissions or entry materials due to system errors, failed, incomplete or garbled computer or other telecommunication transmission malfunctions, hardware or software failures of any kind, lost or unavailable network connections, typographical or system/human errors and failures, technical malfunction(s) of any telephone network or lines, cable connections, satellite transmissions, servers or providers, or computer equipment, traffic congestion on the Internet or at the Competition Website, or any combination thereof, which may limit a Participant’s ability to participate.

#### 16\. RIGHT TO CANCEL, MODIFY OR DISQUALIFY

If for any reason the Competition is not capable of running as planned, including infection by computer virus, bugs, tampering, unauthorized intervention, fraud, technical failures, or any other causes which corrupt or affect the administration, security, fairness, integrity, or proper conduct of the Competition, Competition Sponsor and Kaggle each reserves the right to cancel, terminate, modify or suspend the Competition. Competition Sponsor and Kaggle each further reserves the right to disqualify any Participant who tampers with, undermines, or otherwise damages the submission process or any other part of the Competition or Competition Website. Any attempt by a Participant to deliberately damage any website, including the Competition Website, or undermine the legitimate operation of the Competition is a violation of criminal and civil laws. Should such an attempt be made, Competition Sponsor and Kaggle each reserves the right to disqualify and seek damages, equitable relief, and any available remedy from any such Participant to the fullest extent of the applicable law.

#### 17\. NOT AN OFFER OR CONTRACT OF EMPLOYMENT

Under no circumstances will the entry of a Submission, the awarding of a Prize, or anything in these Rules be construed as an offer or contract of employment with Competition Sponsor or any of the Competition Entities. You acknowledge that you have submitted your Submission voluntarily and not in confidence or in trust. You acknowledge that no confidential, fiduciary, agency, employment or other similar relationship is created between you and Competition Sponsor or any of the Competition Entities by your acceptance of these Rules or your entry of your Submission.

#### 18\. DEFINITIONS

"Competition Data" are the data or datasets available from the Competition Website for the purpose of use in the Competition, including any prototype or executable code provided on the Competition Website. The Competition Data will contain private and public test sets. Which data belongs to which set will not be made available to Participants.

An “Entry” is when a Participant has joined, signed up, or accepted the rules of a competition. Entry is required to make a Submission to a competition.

“Final Submission” is the Submission selected by the user, or automatically selected by Kaggle in the event not selected by the user, that is/are used for final placement on the competition leaderboard.

“Participant” or “Participant User” is an individual who participates in a competition by entering the competition and making a Submission.

“Private Leaderboard” is a ranked display of Participants’ Submission scores against the private test set. The Private Leaderboard determines the final standing in the competition.

“Public Leaderboard” is a ranked display of Participants’ Submission scores against a representative sample of the test data. This leaderboard is visible throughout the competition.

“Sponsor” is responsible for hosting the competition, which includes but is not limited to providing the data for the competition, determining winners, and enforcing competition rules.

“Submission” is anything provided by the Participant to the Sponsor to be evaluated for competition purposes and determine leaderboard position. A Submission may be made as a model, notebook, prediction file, or other format as determined by the Sponsor.

“Team” is one or more Participants participating together in a Kaggle competition, by officially merging together as a Team within the competition platform.