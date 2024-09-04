1 Скопировать пример (директория `ros2`) в директорию `starter` симулятора

2 Подготовить контейнер с ROS 2, MAVROS, OpenCV и cv_bridge
```bash
apptainer build --sandbox onboard onboard-ros2.def
```

Скопировать или переместить его в директорию `containers` симулятора

3 Запустить симулятор
```bash
cd starter
./cluster.sh settings/race_stud_test_settings.json
```

4 Запустить ПО управления дроном по данным с системы локального позиционирования и камеры
```bash
cd ros2
../exec_onboard.sh 1 1 ./onboard.sh
```

Для останова нажать Ctrl+C
