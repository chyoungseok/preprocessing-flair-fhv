import os, glob, h5py
import numpy as np
from tqdm.auto import tqdm

def convert_npy_to_hdf5(
    root_dir,
    list_exclude: list = None,
    list_include: list = None,
    output_path: str = "flair_slice_dataset.h5"
):
    """
    subject 단위 npy → HDF5로 변환 (2D slice 또는 3D volume 모두 지원)

    - 2D: (N, H, W)
    - 3D: (N, C, H, W)

    Args:
        root_dir: subject 폴더들이 모여있는 상위 경로
        list_exclude: 분석에서 제외할 subject ID 리스트
                      (파일 경로에 이 ID가 하나라도 포함되면 제외)
        list_include: 분석에 포함할 subject ID 리스트
                      (파일 경로에 이 ID가 하나라도 포함된 경우만 포함)
        output_path: 변환된 HDF5 파일 저장 경로
    """
    # --- 파일 경로 수집 ---
    img_files = sorted(glob.glob(os.path.join(root_dir, "*", "*AxialSlices_padded.npy")))
    lab_files = sorted(glob.glob(os.path.join(root_dir, "*", "*label_sliceLevel.npy")))
    subj_dirs = [os.path.basename(os.path.dirname(f)) for f in img_files]

    assert len(img_files) == len(lab_files), "이미지와 라벨 파일 수가 다릅니다!"

    # --- include / exclude 필터링 ---
    def use_this_path(path: str) -> bool:
        # 1) include 리스트가 있으면, 그 중 하나라도 path에 포함되어야 함
        if list_include:
            if not any(sid in path for sid in list_include):
                return False
        # 2) exclude 리스트에 있는 ID가 path에 하나라도 포함되면 제외
        if list_exclude:
            if any(sid in path for sid in list_exclude):
                return False
        return True

    before_count = len(img_files)
    filtered = [
        (img_f, lab_f, subj)
        for img_f, lab_f, subj in zip(img_files, lab_files, subj_dirs)
        if use_this_path(img_f)
    ]

    if not filtered:
        raise ValueError("[ERROR] 필터링 후 남은 subject가 없습니다. list_include / list_exclude를 확인하세요.")

    img_files, lab_files, subj_dirs = zip(*filtered)
    img_files, lab_files, subj_dirs = list(img_files), list(lab_files), list(subj_dirs)

    print(f"[INFO] 필터링 전: {before_count}개 → 필터링 후: {len(img_files)}개")

    # --- 총 샘플 수 및 데이터 shape 계산 (shape 정보만 사용) ---
    total_samples, data_shape = 0, None
    for img_f, lab_f in zip(img_files, lab_files):
        imgs = np.load(img_f, mmap_mode="r")  # memmap: 헤더/shape만 즉시 읽고, 데이터는 지연 로딩
        n_samples = imgs.shape[0]
        total_samples += n_samples
        if data_shape is None:
            data_shape = imgs.shape[1:]  # (H, W) or (C, H, W)

    print(f"[INFO] 총 샘플 수: {total_samples}, 데이터 shape={data_shape}")

    # --- HDF5 저장 ---
    with h5py.File(output_path, "w") as f:
        # 이미지 dataset
        if len(data_shape) == 2:   # 2D slice
            dset_img = f.create_dataset(
                "images", shape=(total_samples, *data_shape),
                dtype="float32", compression="gzip", chunks=(1, *data_shape)
            )
        elif len(data_shape) == 3: # 3D volume
            dset_img = f.create_dataset(
                "images", shape=(total_samples, *data_shape),
                dtype="float32", compression="gzip", chunks=(1, *data_shape)
            )
        else:
            raise ValueError(f"지원하지 않는 데이터 shape: {data_shape}")

        # 라벨 dataset
        dset_lab = f.create_dataset(
            "labels", shape=(total_samples,), dtype="int8",
            compression="gzip", chunks=(1024,)
        )

        # subject_id dataset (가변 길이 문자열)
        string_dt = h5py.string_dtype(encoding="utf-8")
        dset_sid = f.create_dataset("subject_ids", shape=(total_samples,), dtype=string_dt)
        
        # slice_index dataset (각 slice의 1~N 인덱스)
        dset_slIdx = f.create_dataset(
            "slice_index",
            shape=(total_samples,),
            dtype="int8",
            compression="gzip",
            chunks=(1024,)
        )

        # 데이터 채우기 (np.load 1회만)
        offset = 0
        for img_f, lab_f, subj in tqdm(zip(img_files, lab_files, subj_dirs), total=len(img_files)):
            imgs = np.load(img_f).astype("float32")
            labs = np.load(lab_f).astype("int8")
            if labs.ndim == 2:
                labs = labs[:, 0]

            n_samples = imgs.shape[0]

            # slice index 생성: 1 ~ n_samples
            sl_idx = np.arange(1, n_samples + 1, dtype="int8")

            # 저장
            dset_img[offset:offset+n_samples] = imgs
            dset_lab[offset:offset+n_samples] = labs
            dset_sid[offset:offset+n_samples] = [subj] * n_samples
            dset_slIdx[offset:offset+n_samples] = sl_idx

            offset += n_samples

    # 저장된 HDF5 key 목록 출력
    with h5py.File(output_path, "r") as f_check:
        print("[INFO] HDF5 keys:", list(f_check.keys()))
    print(f"[완료] 변환된 HDF5 파일 저장: {output_path}")