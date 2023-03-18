

class StimSequence:

    @staticmethod
    def generate_2d(x0, x1, y0, y1, start_time, end_time, period, val, dur=0):
        n = int((end_time - start_time)/period)
        if dur:
            return [[x0, x1, y0, y1, val, dur, start_time + i*period] for i in range(n)]
        else:
            return [[x0, x1, y0, y1, val, start_time + i*period] for i in range(n)]

    @staticmethod
    def generate_3d(x0, x1, y0, y1, z0, z1, start_time, end_time, period, val, dur=0):
        n = int((end_time - start_time)/period)
        if dur:
            return [[x0, x1, y0, y1, z0, z1, val, dur, start_time + i*period] for i in range(n)]
        else:
            return [[x0, x1, y0, y1, z0, z1, val, start_time + i*period] for i in range(n)]
