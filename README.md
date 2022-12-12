# Object Detection Competition CV-13

## 🕵️Members

<table>
    <th colspan=5>📞 TEAM 031</th>
    <tr height="160px">
        <td align="center">
            <a href="https://github.com/LimePencil"><img src="https://avatars.githubusercontent.com/u/71117066?v=4" width="100px;" alt=""/><br /><sub><b>신재영</b></sub></a>
        </td>
        <td align="center">
            <a href="https://github.com/sjz1"><img src="https://avatars.githubusercontent.com/u/68888169?v=4" width="100px;" alt=""/><br /><sub><b>유승종</b></sub></a>
        </td>
        <td align="center">
            <a href="https://github.com/SangJunni"><img src="https://avatars.githubusercontent.com/u/79644050?v=4" width="100px;" alt=""/><br /><sub><b>윤상준</b></sub></a>
        </td>
        <td align="center">
            <a href="https://github.com/lsvv1217"><img src="https://avatars.githubusercontent.com/u/113494991?v=4" width="100px;" alt=""/><br /><sub><b>이성우</b></sub></a>
        </td>
         <td align="center">
            <a href="https://github.com/0seob"><img src="https://avatars.githubusercontent.com/u/29935109?v=4" width="100px;" alt=""/><br /><sub><b>이영섭</b></sub></a>
        </td>
    </tr>
</table>

## 🗑️재활용 품목 분류를 위한 Object Detection
![image](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/cf214eee-d2b9-4ded-b6c3-a87a82bbcb75/7645ad37-9853-4a85-b0a8-f0f151ef05be..png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221212%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221212T081502Z&X-Amz-Expires=86400&X-Amz-Signature=a552adbce6f79ffc92634577ec7e495a0f12ba165a9c83e35a701bdbaec0f2c7&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%227645ad37-9853-4a85-b0a8-f0f151ef05be..png%22&x-id=GetObject)
>바야흐로 대량 생산, 대량 소비의 시대, 우리는 많은 물건이 대량으로 생산되고, 소비되는 시대를 살고 있습니다. 하지만 이러한 문화는 '쓰레기대란','매립지 부족'과 같은 여러 사회문제를 낳고 있습니다.

>분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나입니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용도지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문입니다. 따라서 우리는 사진에서 쓰레기를 Detection하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다. 문제 해결을 위한 데이터셋으로는 일반 쓰레기, 플라스틱, 종이, 유리 등 10종류의 쓰레기가 찍힌 사진 데이터셋이 제공됩니다. 여러분에 의해 만들어진 우수한 성능의 >모델은 쓰레기장에 설치되어 정확환 분리수거를 돕거나, 어린아이들의 분리수거 교육 등에 사용될 수 있을 것입니다. 부디 지구를 위기로부터 구해주세요! 🌎

## 💾 Datasets
- 전체 이미지 개수 : 9754장
   - train : 4883장
   - test : 4871장
- 10 class : General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
- 이미지 크기 : (1024, 1024)
- annotation format : COCO format


## 🗓️Timeline
![images](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/c6704a5b-4508-45ad-8074-d688e392d3c6/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221212%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221212T065649Z&X-Amz-Expires=86400&X-Amz-Signature=fe89efb384fd548b23abf154bffeded6daeaf49ed4e65577bd3cc7e710c601d6&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)

## 🧑‍💻Team Roles
><b>신재영</b>
>
>&nbsp;&nbsp;&nbsp;&nbsp;Ensemble, data cleaning, k-fold 등 다양한 코드 작성
>
>&nbsp;&nbsp;&nbsp;&nbsp;slack-wandb, slack-jira 등 다양한 협업 툴 셋업
>
>&nbsp;&nbsp;&nbsp;&nbsp;baseline실험과 mosaic 실험


> <b>유승종</b>
>
>&nbsp;&nbsp;&nbsp;&nbsp;RGB color EDA
>
>&nbsp;&nbsp;&nbsp;&nbsp;Data Augmentation experiments(Sharpen, Gaussia Noise)
>
>&nbsp;&nbsp;&nbsp;&nbsp;Focal loss hyperparameter tunning


> <b>윤상준</b>
>
>&nbsp;&nbsp;&nbsp;&nbsp;Ensemble, k-fold 등 코드 작성
>
>&nbsp;&nbsp;&nbsp;&nbsp;2-stage model baseline
>
>&nbsp;&nbsp;&nbsp;&nbsp;Number of object for each image, bbox size for each class 


> <b>이성우</b>
>
>&nbsp;&nbsp;&nbsp;&nbsp;Pseudo labeling 코드 작성, mAP metric 분석
>
>&nbsp;&nbsp;&nbsp;&nbsp;Image augmentation, Hyperparameter tunning
>
>&nbsp;&nbsp;&nbsp;&nbsp;Number of object for each class EDA


> <b>이영섭</b>
>
>&nbsp;&nbsp;&nbsp;&nbsp;pseudo labeling 코드 작성, mAP metric 분석
>
>&nbsp;&nbsp;&nbsp;&nbsp;팀 계획 수립
>
>&nbsp;&nbsp;&nbsp;&nbsp;bbox size EDA, Image augmentation experiment, Ensemble
>

## 🏔️Environments
### <img src="https://cdn3.emoji.gg/emojis/4601_github.png" alt="drawing" width="16"/>  GitHub
- 모든 코드들의 버전관리
- GitFlow를 이용한 효율적인 전략
- Issue를 통해 버그나 프로젝트 관련 기록
- PR을 통한 code review
- 총 55개의 PR, 161개의 commit

### <img src="https://img.icons8.com/ios-filled/500/notion.png" alt="drawing" width="16"/> Notion
- 노션을 이용하여 실험결과등을 정리
- 회의록을 매일 기록하여 일정을 관리
- 가설 설정및 결과 분석등을 기록
- 캘린더를 사용하여 주간 일정 관리

### <img src="https://cdn.icon-icons.com/icons2/2699/PNG/512/atlassian_jira_logo_icon_170511.png" alt="drawing" width="16"/> Jira
- 발생하는 모든 실험의 진행상황 기록
- 로드맵을 통한 스케줄 관리
- 효율적인 일 분배 및 일관성 있는 branch 생성
- 총 81개의 Issue 발생

### <img src="https://avatars.githubusercontent.com/u/26401354?s=200&v=4" alt="drawing" width="16"/> WandB
- 실험들의 기록 저장 및 공유
- 모델들의 성능 비교
- Hyperparameter 기록
- 총 500시간 기록

## ⚙️Requirements
```
Ubuntu 18.04.5 LTS
Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz
NVIDIA Tesla V100-PCIE-32GB

conda install pytorch=1.7.1 cudatoolkit=11.0 torchvision -c pytorch  
pip install openmim  
mim install mmdet  
```

## 🎉Results🎉
>### Public LB : 5th (mAP 0.7019)
![images](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/5b818cff-d49f-4aef-928e-7b33124f5f84/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221212%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221212T065252Z&X-Amz-Expires=86400&X-Amz-Signature=d31b84369e2007409ab78dbdfb8ea7d944597175b15a66127f3814bab901d301&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)
>### Private LB : 5th (mAP 0.6871)
![images](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/37782bf4-cb09-4970-b2cf-378e814f9e5f/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221212%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221212T065337Z&X-Amz-Expires=86400&X-Amz-Signature=09b243216325e37595d22b76204c255e577daf81fa61eb73d1b3aeb44db7651f&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)

## 📌Please Look at our Wrap-Up Report for more details
[![image](https://user-images.githubusercontent.com/62556539/200262300-3765b3e4-0050-4760-b008-f218d079a770.png)](https://www.notion.so/Wrap-up-Report-d2fbd966d5cf418483aef20acfc0443e)
