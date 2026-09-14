import React from 'react';
import { Task, Note, Priority } from '../types';

interface AreaDetailsProps {
  area: string;
  color: string;
  tasks: Task[];
  notes: Note[];
  onOpenTask: (task: Task) => void;
  onOpenNote: (note: Note) => void;
  onToggleTask: (taskId: string, completed: boolean) => void;
  onNewTask: () => void;
  onNewNote: () => void;
}

const AreaDetails: React.FC<AreaDetailsProps> = ({ area, color, tasks, notes, onOpenTask, onOpenNote, onToggleTask, onNewTask, onNewNote }) => {
  const priorityMap: Record<string, number> = { [Priority.High]: 3, [Priority.Medium]: 2, [Priority.Low]: 1 };

  // Sort Tasks: Active first, then by Priority, then by CreatedAt DESC
  const sortedTasks = [...tasks].sort((a, b) => {
      if (a.completed !== b.completed) return a.completed ? 1 : -1;
      
      if (!a.completed) {
          const pA = a.priority ? priorityMap[a.priority] : 0;
          const pB = b.priority ? priorityMap[b.priority] : 0;
          if (pA !== pB) return pB - pA;
      }
      
      return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
  });

  // Sort Notes: CreatedAt DESC
  const sortedNotes = [...notes].sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());

  // Use passed color
  const areaColor = color;

  return (
    <div className="px-6 py-8 h-full flex flex-col">
        <header className="mb-8 flex items-center justify-between gap-4">
            <div className="flex items-center gap-4">
                <span className={`w-4 h-4 rounded-full ${areaColor} shadow-md`}></span>
                <h1 className="text-3xl font-bold text-slate-900 dark:text-white">{area}</h1>
            </div>
            <div className="flex items-center gap-3">
                <button onClick={onNewTask} className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium shadow-sm transition-all bg-primary text-white hover:bg-primary-hover shadow-primary/30">
                    <span className="material-icons text-lg text-white">add_task</span>
                    New Task
                </button>
                <button onClick={onNewNote} className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium shadow-sm transition-all bg-white dark:bg-neutral-surface-dark border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300 hover:border-primary hover:text-primary">
                    <span className="material-icons text-lg text-primary">note_add</span>
                    New Note
                </button>
            </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 flex-1 overflow-hidden min-h-0">
            {/* TASKS COLUMN */}
            <div className="flex flex-col min-h-0">
                <div className="flex items-center justify-between mb-4">
                    <h2 className="text-lg font-bold text-slate-700 dark:text-slate-200 flex items-center gap-2">
                        <span className="material-icons text-slate-400">check_circle</span> Tasks
                    </h2>
                    <span className="bg-slate-100 dark:bg-slate-800 px-2 py-0.5 rounded text-xs font-semibold text-slate-500">{tasks.length}</span>
                </div>
                
                <div className="flex-1 overflow-y-auto pr-2 space-y-3 custom-scrollbar">
                    {sortedTasks.map(task => (
                        <div 
                            key={task.id}
                            onClick={() => onOpenTask(task)}
                            className={`group bg-white dark:bg-neutral-surface-dark rounded-xl p-4 border border-slate-200 dark:border-slate-800 shadow-sm hover:shadow-md transition-all cursor-pointer flex items-start gap-4 ${task.completed ? 'opacity-60 bg-slate-50 dark:bg-slate-900/40' : ''}`}
                        >
                             <div data-testid={`toggle-task-${task.id}`} className="pt-1" onClick={(e) => { e.stopPropagation(); onToggleTask(task.id, task.completed); }}>
                                <div className={`w-5 h-5 rounded border flex items-center justify-center transition-colors ${
                                    task.completed 
                                    ? 'bg-slate-300 dark:bg-slate-700 border-transparent text-white' 
                                    : 'border-slate-300 hover:border-primary text-transparent hover:text-primary'
                                }`}>
                                     <span className="material-icons text-sm">{task.completed ? 'check' : 'check'}</span>
                                </div>
                            </div>
                            <div className="flex-1 min-w-0">
                                <div className="flex items-start justify-between">
                                    <h4 className={`font-semibold text-slate-900 dark:text-white truncate ${task.completed ? 'line-through text-slate-500' : ''}`}>{task.title}</h4>
                                    {!task.completed && task.priority && (
                                        <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded uppercase ${
                                            task.priority === Priority.High ? 'bg-red-50 text-red-600' : 
                                            task.priority === Priority.Medium ? 'bg-orange-50 text-orange-600' : 
                                            'bg-blue-50 text-blue-600'
                                        }`}>{task.priority}</span>
                                    )}
                                </div>
                                <p className="text-slate-500 text-sm mt-1 line-clamp-2">{task.description}</p>
                                <div className="mt-2 text-xs text-slate-400">
                                    {task.due_date && new Date(task.due_date).toLocaleDateString()}
                                </div>
                            </div>
                        </div>
                    ))}
                    {tasks.length === 0 && (
                        <div className="text-center py-10 text-slate-400 italic">No tasks yet.</div>
                    )}
                </div>
            </div>

            {/* NOTES COLUMN */}
            <div className="flex flex-col min-h-0">
                 <div className="flex items-center justify-between mb-4">
                    <h2 className="text-lg font-bold text-slate-700 dark:text-slate-200 flex items-center gap-2">
                        <span className="material-icons text-slate-400">description</span> Notes
                    </h2>
                     <span className="bg-slate-100 dark:bg-slate-800 px-2 py-0.5 rounded text-xs font-semibold text-slate-500">{notes.length}</span>
                </div>

                <div className="flex-1 overflow-y-auto pr-2 space-y-4 custom-scrollbar">
                     {sortedNotes.map(note => (
                        <div 
                            key={note.id}
                            onClick={() => onOpenNote(note)}
                            className="bg-white dark:bg-neutral-surface-dark rounded-xl p-5 border border-slate-200 dark:border-slate-800 shadow-sm hover:shadow-md hover:border-primary/30 transition-all cursor-pointer group"
                        >
                            <h3 className="font-bold text-slate-900 dark:text-white mb-2 group-hover:text-primary transition-colors">{note.title}</h3>
                            <p className="text-sm text-slate-500 dark:text-slate-400 line-clamp-3 mb-3">{note.content}</p>
                            <div className="flex items-center text-xs text-slate-400 border-t border-slate-100 dark:border-slate-800 pt-3">
                                 <span className="material-icons text-sm mr-1">event_note</span>
                                 Created: {new Date(note.created_at).toLocaleDateString()}
                            </div>
                        </div>
                     ))}
                      {notes.length === 0 && (
                        <div className="text-center py-10 text-slate-400 italic">No notes yet.</div>
                    )}
                </div>
            </div>
        </div>
    </div>
  );
};

export default AreaDetails;