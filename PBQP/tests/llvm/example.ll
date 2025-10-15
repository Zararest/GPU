; ModuleID = 'example.c'
source_filename = "example.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; Function Attrs: nofree norecurse nosync nounwind memory(none) uwtable
define dso_local double @high_register_pressure() local_unnamed_addr #0 {
  br label %47

1:                                                ; preds = %47
  %2 = fadd double %64, %65
  %3 = fadd double %2, %66
  %4 = fadd double %3, %67
  %5 = fadd double %4, %68
  %6 = fadd double %5, %69
  %7 = fadd double %6, %70
  %8 = fadd double %7, %71
  %9 = fadd double %8, %72
  %10 = fadd double %9, %73
  %11 = fadd double %10, %74
  %12 = fadd double %11, %75
  %13 = fadd double %12, %77
  %14 = fadd double %13, %79
  %15 = extractelement <16 x double> %85, i64 15
  %16 = fadd double %14, %15
  %17 = extractelement <16 x double> %85, i64 14
  %18 = fadd double %16, %17
  %19 = extractelement <16 x double> %85, i64 13
  %20 = fadd double %18, %19
  %21 = extractelement <16 x double> %85, i64 12
  %22 = fadd double %20, %21
  %23 = extractelement <16 x double> %85, i64 11
  %24 = fadd double %22, %23
  %25 = extractelement <16 x double> %85, i64 10
  %26 = fadd double %24, %25
  %27 = extractelement <16 x double> %85, i64 9
  %28 = fadd double %26, %27
  %29 = extractelement <16 x double> %85, i64 8
  %30 = fadd double %28, %29
  %31 = extractelement <16 x double> %85, i64 7
  %32 = fadd double %30, %31
  %33 = extractelement <16 x double> %85, i64 6
  %34 = fadd double %32, %33
  %35 = extractelement <16 x double> %85, i64 5
  %36 = fadd double %34, %35
  %37 = extractelement <16 x double> %85, i64 4
  %38 = fadd double %36, %37
  %39 = extractelement <16 x double> %85, i64 3
  %40 = fadd double %38, %39
  %41 = extractelement <16 x double> %85, i64 2
  %42 = fadd double %40, %41
  %43 = extractelement <16 x double> %85, i64 1
  %44 = fadd double %42, %43
  %45 = extractelement <16 x double> %85, i64 0
  %46 = fadd double %44, %45
  ret double %46

47:                                               ; preds = %0, %47
  %48 = phi i32 [ 0, %0 ], [ %86, %47 ]
  %49 = phi double [ 2.300000e+00, %0 ], [ %79, %47 ]
  %50 = phi double [ 2.200000e+00, %0 ], [ %77, %47 ]
  %51 = phi double [ 2.100000e+00, %0 ], [ %75, %47 ]
  %52 = phi double [ 2.000000e+00, %0 ], [ %74, %47 ]
  %53 = phi double [ 1.900000e+00, %0 ], [ %73, %47 ]
  %54 = phi double [ 1.800000e+00, %0 ], [ %72, %47 ]
  %55 = phi double [ 1.700000e+00, %0 ], [ %71, %47 ]
  %56 = phi double [ 1.600000e+00, %0 ], [ %70, %47 ]
  %57 = phi double [ 1.500000e+00, %0 ], [ %69, %47 ]
  %58 = phi double [ 1.400000e+00, %0 ], [ %68, %47 ]
  %59 = phi double [ 1.300000e+00, %0 ], [ %67, %47 ]
  %60 = phi double [ 1.200000e+00, %0 ], [ %66, %47 ]
  %61 = phi double [ 1.100000e+00, %0 ], [ %65, %47 ]
  %62 = phi double [ 1.000000e+00, %0 ], [ %64, %47 ]
  %63 = phi <16 x double> [ <double 3.900000e+00, double 3.800000e+00, double 3.700000e+00, double 3.600000e+00, double 3.500000e+00, double 3.400000e+00, double 3.300000e+00, double 3.200000e+00, double 3.100000e+00, double 3.000000e+00, double 2.900000e+00, double 2.800000e+00, double 2.700000e+00, double 2.600000e+00, double 2.500000e+00, double 2.400000e+00>, %0 ], [ %85, %47 ]
  %64 = tail call double @llvm.fmuladd.f64(double %62, double %61, double %60)
  %65 = tail call double @llvm.fmuladd.f64(double %61, double %60, double %59)
  %66 = tail call double @llvm.fmuladd.f64(double %60, double %59, double %58)
  %67 = tail call double @llvm.fmuladd.f64(double %59, double %58, double %57)
  %68 = tail call double @llvm.fmuladd.f64(double %58, double %57, double %56)
  %69 = tail call double @llvm.fmuladd.f64(double %57, double %56, double %55)
  %70 = tail call double @llvm.fmuladd.f64(double %56, double %55, double %54)
  %71 = tail call double @llvm.fmuladd.f64(double %55, double %54, double %53)
  %72 = tail call double @llvm.fmuladd.f64(double %54, double %53, double %52)
  %73 = tail call double @llvm.fmuladd.f64(double %53, double %52, double %51)
  %74 = tail call double @llvm.fmuladd.f64(double %52, double %51, double %50)
  %75 = tail call double @llvm.fmuladd.f64(double %51, double %50, double %49)
  %76 = extractelement <16 x double> %63, i64 15
  %77 = tail call double @llvm.fmuladd.f64(double %50, double %49, double %76)
  %78 = extractelement <16 x double> %63, i64 14
  %79 = tail call double @llvm.fmuladd.f64(double %49, double %76, double %78)
  %80 = shufflevector <16 x double> %63, <16 x double> poison, <16 x i32> <i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14>
  %81 = insertelement <16 x double> %80, double %64, i64 0
  %82 = shufflevector <16 x double> %63, <16 x double> poison, <16 x i32> <i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13>
  %83 = insertelement <16 x double> %82, double %65, i64 0
  %84 = insertelement <16 x double> %83, double %64, i64 1
  %85 = tail call <16 x double> @llvm.fmuladd.v16f64(<16 x double> %63, <16 x double> %81, <16 x double> %84)
  %86 = add nuw nsw i32 %48, 1
  %87 = icmp eq i32 %86, 100
  br i1 %87, label %1, label %47, !llvm.loop !5
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fmuladd.f64(double, double, double) #1

; Function Attrs: nofree norecurse nosync nounwind memory(none) uwtable
define dso_local i32 @main() local_unnamed_addr #0 {
  %1 = tail call double @high_register_pressure()
  %2 = fptosi double %1 to i32
  ret i32 %2
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <16 x double> @llvm.fmuladd.v16f64(<16 x double>, <16 x double>, <16 x double>) #2

attributes #0 = { nofree norecurse nosync nounwind memory(none) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{!"Ubuntu clang version 18.1.3 (1ubuntu1)"}
!5 = distinct !{!5, !6}
!6 = !{!"llvm.loop.mustprogress"}
