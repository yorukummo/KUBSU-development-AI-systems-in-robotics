from GUI import GUI
from HAL import HAL
import cv2
import numpy as np 

kp = 0.005  # Пропорциональный коэффициент для PID-регулятора

while True:
    # Получаем изображение с камеры пылесоса
    frame = HAL.getImage()
    
    # Преобразуем изображение из цветового пространства BGR в HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Определяем диапазон красного цвета для фильтрации линии
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    # Создаем маску, чтобы выделить красные участки изображения
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Находим контуры на маске (внешние контуры)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    

    if len(contours) > 0:
        # Из всех контуров выбираем самый большой (с максимальной площадью)
        line_contour = max(contours, key=cv2.contourArea)
        
        # Вычисляем моменты контура
        M = cv2.moments(line_contour)
        
        # Находим центр тяжести (центроид) контура
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # Определяем центр изображения
        image_center = frame.shape[1] // 2
        
        # Вычисляем ошибку: разница между центром линии и центром изображения
        error = cX - image_center
        
        # Рассчитываем управляющий сигнал (скорректированное направление)
        steering = kp * error

        # Рисуем контур линии на изображении
        cv2.drawContours(frame, [line_contour], -1, (0, 255, 0), 2)
        
        # Рисуем круг в точке центра линии
        cv2.circle(frame, (cX, cY), 5, (0, 255, 0), -1)

        # Устанавливаем линейную скорость движения пылесоса
        HAL.setV(1)  

        # Рассчитываем угловую скорость (для поворотов)
        angular_velocity = -steering  # Отрицательное значение для поворота влево, положительное — вправо

        # Устанавливаем угловую скорость пылесоса
        HAL.setW(angular_velocity)

    # Отображаем текущее изображение с наложенными контурами и метками
    GUI.showImage(frame)
