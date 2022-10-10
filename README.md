Реализован cyclegan по статье https://arxiv.org/abs/1703.10593 для переноса стилей изображений.

Модель можно протестировать напрямую  через IDE указав путь к изображению image_path и другие необходимые параметры в train_params и запустив модуль change_image.

Для запуска обучения модели используйте модуль train.

Ввиду недостаточных ресурсов колаба cyclegan обучал для выходного разренения image_size = 128. Как опцию увеличения разрешения попробовал ESRGAN (веса взяты
с https://github.com/xinntao/ESRGAN. интересно было потестить). 

https://t.me/dlsgantestbot - адрес бота бля теста. После отправки изображения, бот стилизует его под картины Ван Гога. Если бот будет офлайн, можно связаться со мной или использовать собственный TOKEN.

Чтобы получить TOKEN отправьте '/start' сообщение боту https://t.me/BotFather или для дополнительной информации - https://core.telegram.org/bots.

Текущие веса cyclegan получены где-то для 60 эпох:

![alt text](https://github.com/HlodM/cyclegan/blob/main/weights/images/bot_image1.jpg?raw=true)
![alt text](https://github.com/HlodM/cyclegan/blob/main/weights/images/vg_image1.jpg?raw=true)
![alt text](https://github.com/HlodM/cyclegan/blob/main/weights/images/bot_image2.jpg?raw=true)
![alt text](https://github.com/HlodM/cyclegan/blob/main/weights/images/vg_image2.jpg?raw=true)
![alt text](https://github.com/HlodM/cyclegan/blob/main/weights/images/bot_image3.jpg?raw=true)
![alt text](https://github.com/HlodM/cyclegan/blob/main/weights/images/vg_image3.jpg?raw=true)
![alt text](https://github.com/HlodM/cyclegan/blob/main/weights/images/bot_image4.jpg?raw=true)
![alt text](https://github.com/HlodM/cyclegan/blob/main/weights/images/vg_image4.jpg?raw=true)

Часть изображений смещается в зеленоватый оттенок. Для борьбы статье было предложено дополнительно ввести лосс на идентичность, который я добавил после 40 эпохи, что позволило частично решить проблему. 
