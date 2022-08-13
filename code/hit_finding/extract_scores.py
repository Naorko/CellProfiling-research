import os

from metrics import *

# threshs = [2, 3, 4, 5, 6, 7, 8, 10, 15, 20]  # 10 Options
threshs = [9, 11, 12, 13, 14, 16, 17, 18, 19]  # 9 Options
plates = os.listdir(f'{zsc_fld}/pure/err')  # 401 plates
plates.sort()

inp = int(sys.argv[1])
plate_slice_id = inp % 100
f = 4

start_id = plate_slice_id * f
end_id = start_id + f
end_id = len(plates) if len(plates) - end_id < 3 else end_id
plates = plates[start_id:end_id]

t_id = inp // 100
thresh = threshs[t_id]

out_path = f'{zsc_fld}/frac_score_new/{thresh}'
os.makedirs(out_path, exist_ok=True)

for p in plates:
    print(p)
    df_res = extract_multiple_scores(extract_score, p, threshold=thresh)
    # print(os.path.join(out_path, p))
    df_res.to_csv(os.path.join(out_path, p))
