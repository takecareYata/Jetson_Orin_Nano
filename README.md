# Jetson_Orin_Nano
### 1. install python3.10-venv on ubuntu
> version은 현재 python version을 확인하고 결정

```bash
sudo apt install python3.10-venv
```

### 2. working directory에 venv 설정

* `work` directory를 만들고 change directory

    ```bash
    mkdir work && cd work
    ```

* 다음 명령으로 `venv` 이름으로 virtual evironment setup

    `--system-site-packages` option을 이용하여 현재 jetpack version 환경을 그대로 사용

    ```bash
    python -m venv --system-site-packages venv
    ```

* activate `venv`

    ```bash
    source ~/work/venv/bin/activate
    ```

    **prompt**가 아래와 같이 바뀌었는지 확인

    ```sh
    (venv) aidl@jetson-200:~$
    ```

### 3. VS code `remote ssh` 연결로 jetson board 개발 환경 구축

* VS code에서 새창 열기를 진행한다.
* Remote-ssh 확장 pack이 install 되어 있는지 확인하고 없으면 install
* ssh 연결 추가를 선택하고 아래와 입력하고 `.ssh/config`를 확인한다.

    ```text
    ssh -Y YOURADDRESS
    ```

* SSH list 에서 연결할 보드를 선택하고 현재 창에서 연결을 진행하다.

> 1. 보드의 system을 `linux`로 선택한다.
> 2. passwd를 입력하고 진행한다.
> 3. `EXPLORER` 창에서 `Open Floder`를 선택하여 jetson의 workding folder를 연다.

* 연결이 되었으면 필요한 파일들을 working space 에 `drag & drop`으로 copy 한다.

> 1. `examples_jetson_260530.zip`을 unzip 한 `example` folder
> 2. `utils.zip`을 unzip 하고 `utils/pip_packages/requirements.txt` file

* copy 완료 후 terminal 창에서 `venv`를 activate 하고 `work` directory 안에서 다음 명령을 실행한다.

    ```bash
    pip install –r requirements.txt
    ```

---

## `ultralytics` 와 `pytorch` 설치

ultralytics YOLO를 사용하기 위해서 pytorch를 install

### 1. copy pytorch `.whl` file to board

* unzip한 `utils/torch_whl` folder를 board에 copy
* `torch_whl` folder에서 다음 명령을 수행하여 torch를 install

```bash
pip install --no-cache torch-2.3.0-cp310-cp310-linux_aarch64.whl 
pip install --no-cache torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl 
pip install --no-cache torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl
```

### 2.`ultralytics` install

* 다음 명령을 수행하여 numpy와 pytorch version을 유지하고 install

```bash
pip install ultralytics 'numpy==1.26.4' 'torch==2.3'
```