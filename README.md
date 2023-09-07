# Text Generation Model based on Truyen Kieu (The Tale of Kieu)

## Introduction

This repository contains a text generation model that is based on text data with LSTM and dense layers. The model is trained on a large corpus of text, such as <b>The Tale of Kieu</b>.

The model is implemented in the Python programming language using the Keras deep learning library. The model is made up of two layers: an LSTM layer and a dense layer. The LSTM layer is responsible for learning the long-term dependencies in the text data. The dense layer is responsible for learning the relationships between the input and output features of the model.

## Model
```
x = inputs
x = self.embedding(x, training=training)
# LSTM 1
if states == None:
    state_h, state_c = self.lstm_0.get_initial_state(x)
    states = [state_h, state_c]
x, state_h, state_c = self.lstm_0(x, initial_state=states, training=training)
# FC
x = self.dense_0(x, training=training)
x = self.dropout_0(x)
x = self.dense_1(x, training=training)
x = self.dropout_1(x)
x = self.dense_2(x, training=training)

if return_state:
    return x, [state_h, state_c]
else:
    return x
```

## Preview
Text to speech:
https://github.com/thanhtung4work/ThuyKieuGPT/assets/67087253/74217e2f-1b30-4ecd-b498-640ad82073d6

```
Starting string: này em
Number of line 50
```

```
này em
trông may người bốn tiến dao
bấn lầu giến bóng trước âu bên thân
khuy tiên giậc có lu thân
phận đàm mừa cũng chi phu thế là la
ngận rằng càng nhớc ngoái thâ
một ngàng ngày nhớ bương nhâm tửa tình
lạc thiền hương lộn la nhi
nhà đôi thở tiếp va nhàng bử thân
cũnh thì lụi gác vui từa
cũng lài ngọc những biệu là đền sau
còn lầu khác giấc mưa đào
lửa cho thấy mặt là mày sao chưa
néo chàng nướng một người ngay
bạc chà đã thấy còn là khúc nhay
sinh thi đã buổ lâm thầy
duyên pho nào có gie say có thi
liền bao mách liết riền mai
thàng màng nghi đã ngong ngan chầm dần
tiền thâu trưới dứ ginh ngày
giến hàng bào một vài người lời th
người ra bọn có khuy và
tử bương phải hoá ngườ ta b đ như
tiều bào khi nhụ ngói trời
tiếng ba mội lạc bươi chiếu them
thoa khi trăng mã tóng thòng
bấy này có vậc lòn trần chưa cha
sống thân quyến cánh ngầy
tưởng đề đổi tật mặt mày thương trôi
nửa chồng thủ lạc thương mồn
ngắc trần mình nặt cánh trần tưởng đên
thoắt luống đổi buôn điềnh
tiễu thì đã nét tao tgười cin trơi
bảnh rằng: sát đế chọt đời
góp tay nhà cũng có người ở đân
thế biết lâu đã và cây
còn xong đó nhớ mặt vày chy ngay
lệ nhờ còn đóa càn cha
phại tưần một chúc mặt nhà lành xan
đước trần mường cũng tanh thên
khốc thâm tười cũng miếng là là đâu
thuyên ưa cương lới một nhà
dvới nhùng như cũng trên vài thẳng the
sự lời thệc lạc sanh phà
nghĩ trong nhà cũng thiếu là hong đây
chẳng xưa buồng nhạn xuến la
càng mừa trút lét sao là làng nhừa
tương trời vùng để nao tơ
miết rai nhìn mắt đoóng này lấm thiên
bân thây buộn hén hay xuyền
```
