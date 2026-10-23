#ifndef __CONFIG_H__
#define __CONFIG_H__

// 发布项目，发布前必须确认为1
#define APP_RELEASE                  1

#if APP_RELEASE

// 自动发送数据的间隔时间，单位为1毫秒，最小值为10
#define AUTO_SEND_TIMEOUT            40


/* 小车功能开关 */
#define ENABLE_LOW_BATTERY_ALARM     1
#define ENABLE_CLEAR_RXBUF           1
#define ENABLE_CHECKSUM              1
#define ENABLE_IWDG                  0
#define ENABLE_KEY_RELEASE           1
#define ENABLE_OLED                  1
#define ENABLE_CAR_SUNRISE_ONLY      0


#define ENABLE_ICM20948              1
#define ENABLE_MPU9250               1
#define ENABLE_IMU_ERROR_PASS        0
#define ENABLE_YAW_ADJUST            1
#define ENABLE_ROLL_PITCH            0
#define ENABLE_REAL_WHEEL            0


/* 功能保存Flash */
#define ENABLE_RESET_FLASH           0
#define ENABLE_FLASH                 1
#define ENABLE_AUTO_REPORT           1


/* 打印调试的功能 */
#define ENABLE_DEBUG_ICM_ATT         0
#define ENABLE_DEBUG_MPU_ATT         0
#define ENABLE_DEBUG_SBUS            0
#define ENABLE_DEBUG_YAW             0
#define ENABEL_DEBUG_ENCODER         0


#else /* 调试内容在此处修改，发布时不参与编译 */

// 自动发送数据的间隔时间，单位为1毫秒，最小值为10
#define AUTO_SEND_TIMEOUT            40


/* 小车功能开关 */
#define ENABLE_LOW_BATTERY_ALARM     1
#define ENABLE_CLEAR_RXBUF           1
#define ENABLE_CHECKSUM              1
#define ENABLE_IWDG                  0
#define ENABLE_KEY_RELEASE           0
#define ENABLE_OLED                  1
#define ENABLE_CAR_SUNRISE_ONLY      0


#define ENABLE_ICM20948              0
#define ENABLE_MPU9250               0
#define ENABLE_IMU_ERROR_PASS        0
#define ENABLE_YAW_ADJUST            1
#define ENABLE_ROLL_PITCH            1
#define ENABLE_REAL_WHEEL            0


/* 功能保存Flash */
#define ENABLE_RESET_FLASH           0
#define ENABLE_FLASH                 0
#define ENABLE_AUTO_REPORT           0


/* 打印调试的功能 */
#define ENABLE_DEBUG_ICM_ATT         0
#define ENABLE_DEBUG_MPU_ATT         0
#define ENABLE_DEBUG_SBUS            0
#define ENABLE_DEBUG_YAW             0
#define ENABEL_DEBUG_ENCODER         0



#endif /* APP_RELEASE */


/* ==================== 方向校准（纯软件纠正接线方向）====================
 *  1  = 保持原方向
 *  -1 = 反向
 *
 *  判定方法见《编码器方向修正说明.md》：
 *    架空 -> 手转轮子往车头方向 -> 编码器计数应增大
 *    增大填 1，减小填 -1，始终不动 = 没信号（不是改这里）
 */
#define MOTOR_DIR_M1                (1)
#define MOTOR_DIR_M2                (1)
#define MOTOR_DIR_M3                (1)
#define MOTOR_DIR_M4                (1)

#define ENCODER_DIR_M1              (1)
#define ENCODER_DIR_M2              (-1)
#define ENCODER_DIR_M3              (1)
#define ENCODER_DIR_M4              (-1)

/* 阿克曼转向舵机方向：1 = 角度增大向右转，-1 = 反向 */
#define AKM_STEER_DIR               (1)


#endif /* __CONFIG_H__ */
