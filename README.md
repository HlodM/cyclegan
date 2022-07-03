https://t.me/dlsgantestbot - адрес бота бля теста. после отправки 
изображения, бот стилизует его под картины Ван Гога.

Предполагается, что бот будет хоститься на моем компе. Но, так как
вероятность совпасть по времени не высокая, то в качестве варианта для проверки 
можете воспользоваться вашим токеном.

Чтобы получить TOKEN отправьте '/start' сообщение боту https://t.me/BotFather или
для дополнительной информации - https://core.telegram.org/bots

Или можно протестировать модель напрямую  через IDE указав путь к изображению image_path
в train_params и запустив модуль change_image

Ввиду недостаточных ресурсов колаба cyclegan обучал для выходного разренения
image_size = 128. Как опцию увеличения разрешения попробовал ESRGAN (не обучал, а просто
скопипастил и скачал веса. интересно было потестить). 

На момент коммита, веса cyclegan получены где-то для 40 эпох.
Но промежуточные результаты уже видны:

![alt text](https://github.com/HlodM/cyclegan/blob/main/weights/images/bot_image1.jpg?raw=true)
![alt text](https://github.com/HlodM/cyclegan/blob/main/weights/images/vg_image1.jpg?raw=true)
![alt text](https://github.com/HlodM/cyclegan/blob/main/weights/images/bot_image4.jpg?raw=true)
![alt text](https://github.com/HlodM/cyclegan/blob/main/weights/images/vg_image4.jpg?raw=true)


статься: https://arxiv.o
rg/abs/1703.10593