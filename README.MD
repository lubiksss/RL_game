## *참고
<a href='https://lubiksss.github.io/ml/RL_cartpole/'>DQN을 통한 cartpole게임 학습</a>

간단한 DQN모델을 통해서 CartPole게임을 pytorch를 통해 구현했습니다. 바로 아래 학습 결과를 CartPole이 쓰러지지 않고 버티는걸 볼 수 있습니다.  

DQN을 구현하는데 환경은 gym을 통해서 얻었고, NN에는 Bellman Equation, Replay Buffer, Double Deep Q Learning 개념이 사용되었습니다.

## 학습 결과
<div style = 'column-count :2;'>
<p>학습 전</p>
<img src = 'https://user-images.githubusercontent.com/67966414/128750030-1efe42b4-9d27-4ebd-aef4-b2d92dc24253.gif' alt = '학습 전' style="margin-left: auto; margin-right: auto; display: block;">
<p>학습 후</p>
<img src = 'https://user-images.githubusercontent.com/67966414/128750009-61e1297e-1fcc-423b-a314-765f83a01db3.gif' alt = '학습 후' style="margin-left: auto; margin-right: auto; display: block;">
</div>