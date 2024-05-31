class FrameInfo:
    def __init__(
        self,
        frame,
        ball_in_frame,
        ball=(0, 0),
        ball_color=(0, 0, 0),
        ball_lost_tracking=False,
    ):
        self.frame = frame
        self.ball_in_frame = ball_in_frame
        self.ball = ball
        self.ball_color = ball_color
        self.ball_lost_tracking = ball_lost_tracking
