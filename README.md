https://t.me/dlsgantestbot - адрес бота бля теста. После отправки 
изображения, бот стилизует его под картины Ван Гога.

Предполагается, что бот будет хоститься на моем компе. Но, так как
вероятность совпасть по времени не высокая, то в качестве варианта для проверки 
можете воспользоваться вашим токеном.

Чтобы получить TOKEN отправьте '/start' сообщение боту https://t.me/BotFather или
для дополнительной информации - https://core.telegram.org/bots.

Также можно протестировать модель напрямую  через IDE указав путь к изображению image_path и 
другие необходимые параметры в train_params и запустив модуль change_image.

Ввиду недостаточных ресурсов колаба cyclegan обучал для выходного разренения
image_size = 128. Как опцию увеличения разрешения попробовал ESRGAN (не обучал, а просто
скопипастил и скачал веса с https://github.com/xinntao/ESRGAN. интересно было потестить). 

На момент коммита, веса cyclegan получены где-то для 60 эпох.
Но промежуточные результаты уже видны:

![alt text](https://github.com/HlodM/cyclegan/blob/main/weights/images/bot_image1.jpg?raw=true)
![alt text](https://github.com/HlodM/cyclegan/blob/main/weights/images/vg_image1.jpg?raw=true)
![alt text](https://github.com/HlodM/cyclegan/blob/main/weights/images/bot_image2.jpg?raw=true)
![alt text](https://github.com/HlodM/cyclegan/blob/main/weights/images/vg_image2.jpg?raw=true)
![alt text](https://github.com/HlodM/cyclegan/blob/main/weights/images/bot_image3.jpg?raw=true)
![alt text](https://github.com/HlodM/cyclegan/blob/main/weights/images/vg_image3.jpg?raw=true)
![alt text](https://github.com/HlodM/cyclegan/blob/main/weights/images/bot_image4.jpg?raw=true)
![alt text](https://github.com/HlodM/cyclegan/blob/main/weights/images/vg_image4.jpg?raw=true)

Часть изображений смещается в зеленоватый оттенок, в статье упоминалось про это, но обратил внимание не 
сразу. Для борьбы было предложено дополнительно ввести лосс на идентичность. Я его добавил после
40 эпохи, что позволило частично решить проблему. Так же на некоторых изображениях могут появляться
небольшие артефакты, возможно если бы успел провести весь процесс обучения (200 эпох, на колабе 1 эпоха
считалась около 30 минут), то удалось бы минимизировать количество, хотя у авторов тоже места проявлялись.


статья: https://arxiv.org/abs/1703.10593