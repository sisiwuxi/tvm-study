; ModuleID = 'loop_distribute.c'
source_filename = "loop_distribute.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

@.str = private unnamed_addr constant [3 x i8] c"%d\00", align 1

; Function Attrs: nounwind uwtable
define dso_local i32 @llvm_loop_distribution() local_unnamed_addr #0 !dbg !6 {
  %1 = alloca [1280 x i32], align 16
  %2 = alloca [1280 x i32], align 16
  %3 = alloca [1280 x i32], align 16
  %4 = bitcast [1280 x i32]* %1 to i8*, !dbg !8
  call void @llvm.lifetime.start.p0i8(i64 5120, i8* nonnull %4) #5, !dbg !8
  %5 = bitcast [1280 x i32]* %2 to i8*, !dbg !8
  call void @llvm.lifetime.start.p0i8(i64 5120, i8* nonnull %5) #5, !dbg !8
  %6 = bitcast [1280 x i32]* %3 to i8*, !dbg !8
  call void @llvm.lifetime.start.p0i8(i64 5120, i8* nonnull %6) #5, !dbg !8
  br label %7, !dbg !9

7:                                                ; preds = %7, %0
  %8 = phi i64 [ 0, %0 ], [ %13, %7 ]
  %9 = call i32 @rand() #5, !dbg !10
  %10 = getelementptr inbounds [1280 x i32], [1280 x i32]* %2, i64 0, i64 %8, !dbg !11
  store i32 %9, i32* %10, align 4, !dbg !12, !tbaa !13
  %11 = call i32 @rand() #5, !dbg !17
  %12 = getelementptr inbounds [1280 x i32], [1280 x i32]* %3, i64 0, i64 %8, !dbg !18
  store i32 %11, i32* %12, align 4, !dbg !19, !tbaa !13
  %13 = add nuw nsw i64 %8, 1, !dbg !20
  %14 = icmp eq i64 %13, 1280, !dbg !21
  br i1 %14, label %15, label %7, !dbg !9, !llvm.loop !22

15:                                               ; preds = %7, %15
  %16 = phi i64 [ %22, %15 ], [ 0, %7 ]
  %17 = getelementptr inbounds [1280 x i32], [1280 x i32]* %1, i64 0, i64 %16, !dbg !24
  %18 = trunc i64 %16 to i32, !dbg !25
  store i32 %18, i32* %17, align 4, !dbg !25, !tbaa !13
  %19 = getelementptr inbounds [1280 x i32], [1280 x i32]* %2, i64 0, i64 %16, !dbg !26
  %20 = load i32, i32* %19, align 4, !dbg !26, !tbaa !13
  %21 = add nsw i32 %20, 2, !dbg !27
  store i32 %21, i32* %19, align 4, !dbg !28, !tbaa !13
  %22 = add nuw nsw i64 %16, 1, !dbg !29
  %23 = icmp eq i64 %22, 1280, !dbg !30
  br i1 %23, label %24, label %15, !dbg !31, !llvm.loop !32

24:                                               ; preds = %15
  %25 = getelementptr [1280 x i32], [1280 x i32]* %3, i64 0, i64 -1, !dbg !31
  %26 = load i32, i32* %25, align 4
  br label %27, !dbg !31

27:                                               ; preds = %24, %27
  %28 = phi i32 [ %26, %24 ], [ %30, %27 ]
  %29 = phi i64 [ 0, %24 ], [ %32, %27 ]
  %30 = add nsw i32 %28, 3, !dbg !34
  %31 = getelementptr inbounds [1280 x i32], [1280 x i32]* %3, i64 0, i64 %29, !dbg !35
  store i32 %30, i32* %31, align 4, !dbg !36, !tbaa !13
  %32 = add nuw nsw i64 %29, 1, !dbg !29
  %33 = icmp eq i64 %32, 1280, !dbg !30
  br i1 %33, label %34, label %27, !dbg !31, !llvm.loop !32

34:                                               ; preds = %27, %34
  %35 = phi i64 [ %45, %34 ], [ 0, %27 ]
  %36 = getelementptr inbounds [1280 x i32], [1280 x i32]* %1, i64 0, i64 %35, !dbg !37
  %37 = load i32, i32* %36, align 4, !dbg !37, !tbaa !13
  %38 = call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i64 0, i64 0), i32 %37), !dbg !38
  %39 = getelementptr inbounds [1280 x i32], [1280 x i32]* %2, i64 0, i64 %35, !dbg !39
  %40 = load i32, i32* %39, align 4, !dbg !39, !tbaa !13
  %41 = call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i64 0, i64 0), i32 %40), !dbg !40
  %42 = getelementptr inbounds [1280 x i32], [1280 x i32]* %3, i64 0, i64 %35, !dbg !41
  %43 = load i32, i32* %42, align 4, !dbg !41, !tbaa !13
  %44 = call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i64 0, i64 0), i32 %43), !dbg !42
  %45 = add nuw nsw i64 %35, 1, !dbg !43
  %46 = icmp eq i64 %45, 1280, !dbg !44
  br i1 %46, label %47, label %34, !dbg !45, !llvm.loop !46

47:                                               ; preds = %34
  call void @llvm.lifetime.end.p0i8(i64 5120, i8* nonnull %6) #5, !dbg !48
  call void @llvm.lifetime.end.p0i8(i64 5120, i8* nonnull %5) #5, !dbg !48
  call void @llvm.lifetime.end.p0i8(i64 5120, i8* nonnull %4) #5, !dbg !48
  ret i32 0, !dbg !49
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: nounwind
declare dso_local i32 @rand() local_unnamed_addr #2

; Function Attrs: nofree nounwind
declare dso_local i32 @printf(i8* nocapture readonly, ...) local_unnamed_addr #3

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: nounwind readnone uwtable
define dso_local i32 @pragma_loop_distribution() local_unnamed_addr #4 !dbg !50 {
  %1 = alloca [1280 x i32], align 16
  %2 = getelementptr inbounds [1280 x i32], [1280 x i32]* %1, i64 0, i64 0
  %3 = alloca [1280 x i32], align 16
  %4 = bitcast [1280 x i32]* %1 to i8*, !dbg !51
  call void @llvm.lifetime.start.p0i8(i64 5120, i8* nonnull %4) #5, !dbg !51
  %5 = bitcast [1280 x i32]* %3 to i8*, !dbg !51
  call void @llvm.lifetime.start.p0i8(i64 5120, i8* nonnull %5) #5, !dbg !51
  br label %6, !dbg !52

6:                                                ; preds = %6, %0
  %7 = phi i64 [ 0, %0 ], [ %10, %6 ]
  %8 = getelementptr inbounds [1280 x i32], [1280 x i32]* %1, i64 0, i64 %7, !dbg !53
  %9 = trunc i64 %7 to i32, !dbg !54
  store i32 %9, i32* %8, align 4, !dbg !54, !tbaa !13
  %10 = add nuw nsw i64 %7, 1, !dbg !55
  %11 = getelementptr inbounds [1280 x i32], [1280 x i32]* %3, i64 0, i64 %7, !dbg !56
  %12 = trunc i64 %10 to i32, !dbg !57
  store i32 %12, i32* %11, align 4, !dbg !57, !tbaa !13
  %13 = icmp eq i64 %10, 1280, !dbg !58
  br i1 %13, label %14, label %6, !dbg !52, !llvm.loop !59

14:                                               ; preds = %6
  %15 = load i32, i32* %2, align 16
  br label %16, !dbg !61

16:                                               ; preds = %14, %16
  %17 = phi i32 [ %15, %14 ], [ %21, %16 ]
  %18 = phi i64 [ 0, %14 ], [ %22, %16 ]
  %19 = getelementptr inbounds [1280 x i32], [1280 x i32]* %3, i64 0, i64 %18, !dbg !62
  %20 = load i32, i32* %19, align 4, !dbg !62, !tbaa !13
  %21 = add nsw i32 %20, %17, !dbg !63
  %22 = add nuw nsw i64 %18, 1, !dbg !64
  %23 = getelementptr inbounds [1280 x i32], [1280 x i32]* %1, i64 0, i64 %22, !dbg !65
  store i32 %21, i32* %23, align 4, !dbg !66, !tbaa !13
  %24 = icmp eq i64 %22, 1280, !dbg !67
  br i1 %24, label %25, label %16, !dbg !61, !llvm.loop !68

25:                                               ; preds = %16
  %26 = getelementptr inbounds [1280 x i32], [1280 x i32]* %1, i64 0, i64 8, !dbg !71
  %27 = load i32, i32* %26, align 16, !dbg !71, !tbaa !13
  call void @llvm.lifetime.end.p0i8(i64 5120, i8* nonnull %5) #5, !dbg !72
  call void @llvm.lifetime.end.p0i8(i64 5120, i8* nonnull %4) #5, !dbg !72
  ret i32 %27, !dbg !73
}

; Function Attrs: nounwind readnone uwtable
define dso_local i32 @main() local_unnamed_addr #4 !dbg !74 {
  ret i32 0, !dbg !75
}

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind willreturn }
attributes #2 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nofree nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind readnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 10.0.0-4ubuntu1 ", isOptimized: true, runtimeVersion: 0, emissionKind: NoDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "loop_distribute.c", directory: "/home/sisi/git/sisiwuxi/tvm-study/loop_nest")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{!"clang version 10.0.0-4ubuntu1 "}
!6 = distinct !DISubprogram(name: "llvm_loop_distribution", scope: !1, file: !1, line: 5, type: !7, scopeLine: 5, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !2)
!8 = !DILocation(line: 7, column: 3, scope: !6)
!9 = !DILocation(line: 9, column: 3, scope: !6)
!10 = !DILocation(line: 10, column: 12, scope: !6)
!11 = !DILocation(line: 10, column: 5, scope: !6)
!12 = !DILocation(line: 10, column: 10, scope: !6)
!13 = !{!14, !14, i64 0}
!14 = !{!"int", !15, i64 0}
!15 = !{!"omnipotent char", !16, i64 0}
!16 = !{!"Simple C/C++ TBAA"}
!17 = !DILocation(line: 11, column: 12, scope: !6)
!18 = !DILocation(line: 11, column: 5, scope: !6)
!19 = !DILocation(line: 11, column: 10, scope: !6)
!20 = !DILocation(line: 9, column: 19, scope: !6)
!21 = !DILocation(line: 9, column: 14, scope: !6)
!22 = distinct !{!22, !9, !23}
!23 = !DILocation(line: 12, column: 3, scope: !6)
!24 = !DILocation(line: 14, column: 5, scope: !6)
!25 = !DILocation(line: 14, column: 10, scope: !6)
!26 = !DILocation(line: 15, column: 16, scope: !6)
!27 = !DILocation(line: 15, column: 14, scope: !6)
!28 = !DILocation(line: 15, column: 10, scope: !6)
!29 = !DILocation(line: 13, column: 19, scope: !6)
!30 = !DILocation(line: 13, column: 14, scope: !6)
!31 = !DILocation(line: 13, column: 3, scope: !6)
!32 = distinct !{!32, !31, !33}
!33 = !DILocation(line: 17, column: 3, scope: !6)
!34 = !DILocation(line: 16, column: 14, scope: !6)
!35 = !DILocation(line: 16, column: 5, scope: !6)
!36 = !DILocation(line: 16, column: 10, scope: !6)
!37 = !DILocation(line: 19, column: 18, scope: !6)
!38 = !DILocation(line: 19, column: 5, scope: !6)
!39 = !DILocation(line: 20, column: 18, scope: !6)
!40 = !DILocation(line: 20, column: 5, scope: !6)
!41 = !DILocation(line: 21, column: 18, scope: !6)
!42 = !DILocation(line: 21, column: 5, scope: !6)
!43 = !DILocation(line: 18, column: 19, scope: !6)
!44 = !DILocation(line: 18, column: 14, scope: !6)
!45 = !DILocation(line: 18, column: 3, scope: !6)
!46 = distinct !{!46, !45, !47}
!47 = !DILocation(line: 22, column: 3, scope: !6)
!48 = !DILocation(line: 24, column: 1, scope: !6)
!49 = !DILocation(line: 23, column: 3, scope: !6)
!50 = distinct !DISubprogram(name: "pragma_loop_distribution", scope: !1, file: !1, line: 26, type: !7, scopeLine: 26, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!51 = !DILocation(line: 29, column: 3, scope: !50)
!52 = !DILocation(line: 30, column: 3, scope: !50)
!53 = !DILocation(line: 31, column: 5, scope: !50)
!54 = !DILocation(line: 31, column: 10, scope: !50)
!55 = !DILocation(line: 32, column: 13, scope: !50)
!56 = !DILocation(line: 32, column: 5, scope: !50)
!57 = !DILocation(line: 32, column: 10, scope: !50)
!58 = !DILocation(line: 30, column: 14, scope: !50)
!59 = distinct !{!59, !52, !60}
!60 = !DILocation(line: 35, column: 3, scope: !50)
!61 = !DILocation(line: 37, column: 3, scope: !50)
!62 = !DILocation(line: 38, column: 21, scope: !50)
!63 = !DILocation(line: 38, column: 19, scope: !50)
!64 = !DILocation(line: 38, column: 8, scope: !50)
!65 = !DILocation(line: 38, column: 5, scope: !50)
!66 = !DILocation(line: 38, column: 12, scope: !50)
!67 = !DILocation(line: 37, column: 14, scope: !50)
!68 = distinct !{!68, !61, !69, !70}
!69 = !DILocation(line: 40, column: 3, scope: !50)
!70 = !{!"llvm.loop.distribute.enable", i1 true}
!71 = !DILocation(line: 41, column: 10, scope: !50)
!72 = !DILocation(line: 42, column: 1, scope: !50)
!73 = !DILocation(line: 41, column: 3, scope: !50)
!74 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 44, type: !7, scopeLine: 44, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!75 = !DILocation(line: 47, column: 3, scope: !74)
