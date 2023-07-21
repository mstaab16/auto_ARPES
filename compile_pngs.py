def ffmpeg_compile_pngs_to_mp4():
    import os
    # os.system("ffmpeg -framerate 16.6 -pattern_type glob -i 'movie/*.png' -c:v libx264 -pix_fmt yuv420p out.mp4")
    os.system("ffmpeg -framerate 16.6 -i 'movie/%04d.png' -c:v libx264 -pix_fmt yuv420p out.mp4")

if __name__ == '__main__':
    ffmpeg_compile_pngs_to_mp4()
